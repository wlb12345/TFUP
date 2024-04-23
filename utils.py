import clip
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from dassl.data.datasets import Datum
from tqdm import tqdm


CUSTOM_TEMPLATES = {
    "OfficeHome": "a photo of a {}.",
    "VisDA17": "a photo of a {}.",
    "Office31": "a photo of a {}.",
    "DomainNet": "a photo of a {}.",
}


def cls_acc(output, target, topk=1):
    pred = output.topk(topk, 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    acc = float(correct[: topk].reshape(-1).float().sum(0, keepdim=True).cpu().numpy())
    acc = 100 * acc / target.shape[0]
    return acc


class SmoothCrossEntropy(nn.Module):
    def __init__(self, alpha=0.0):
        super(SmoothCrossEntropy, self).__init__()
        self.alpha = alpha

    def forward(self, logits, labels):
        num_classes = logits.shape[-1]
        alpha_div_k = self.alpha / num_classes
        target_probs = F.one_hot(labels, num_classes=num_classes).float() * \
            (1. - self.alpha) + alpha_div_k
        loss = -(target_probs * torch.log_softmax(logits, dim=-1)).sum(dim=-1)
        return loss.mean()


def marginal_loss(logits):
    softmax_out = nn.Softmax(dim=1)(logits)
    msoftmax = softmax_out.mean(dim=0)
    gentropy_loss = torch.sum(-msoftmax * torch.log(msoftmax + 1e-6))
    marginal__loss = -gentropy_loss
    return marginal__loss


class TextAdapter(nn.Module):
    def __init__(self, clip_weights, ratio_beta):
        super(TextAdapter, self).__init__()
        self.feat_dim, self.cate_num = clip_weights.shape
        self.ratio_beta = ratio_beta
        self.res = nn.Parameter(torch.zeros([self.cate_num, self.feat_dim]).half().cuda(), requires_grad=True)
        
    def forward(self, cache_keys, clip_weights, cache_values):
        new_cache_keys = cache_keys
        res_text = self.res.t()
        new_clip_weights = (1 - self.ratio_beta) * clip_weights + self.ratio_beta * res_text
        new_clip_weights = new_clip_weights / new_clip_weights.norm(dim=0, keepdim=True)
        new_cache_keys = new_cache_keys / new_cache_keys.norm(dim=-1, keepdim=True)
        new_cache_values = cache_values
       
        return new_cache_keys.half(), new_clip_weights.half(), new_cache_values.half()


class Adapter(nn.Module):
    def __init__(self, c_in, reduction=4):
        super(Adapter, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(c_in, c_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c_in // reduction, c_in, bias=False),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.fc(x)
        return x


def refine_few_shots_without_label(classnames, text_features, cls_tokens, test_label, data, after_shots, cache_dir,
                                   cache=True):
    class_num = len(classnames)
    cls_token = cls_tokens / cls_tokens.norm(dim=-1, keepdim=True)
    text_feature = text_features / text_features.norm(dim=-1, keepdim=True)
    if not cache:
        alpha = 0.3
        beta = 1
        cache_keys = torch.load(cache_dir+'/cache_keys.pt')
        cache_values = torch.load(cache_dir+'/cache_values.pt')
        cache_values = cache_values.long()
        cache_values = F.one_hot(cache_values).half()
        cache_keys = cache_keys / cache_keys.norm(dim=-1, keepdim=True)
        R_fF = cls_token @ cache_keys.t()
        sim = get_sim_based_kl(cache_keys, cls_token, text_feature.t())
        R_fF = R_fF * sim
        cache_logits = ((-1) * (beta - beta * R_fF)).exp() @ cache_values
        R_fW = 100. * cls_token @ text_feature.t()
        cache_logits = R_fW + alpha * cache_logits
        pseudo_label = cache_logits.max(dim=1).indices
    logits = F.softmax(10. * cls_token @ text_feature.t(), dim=1)
    cls_tokens_after = []
    label_after = []
    true_label = []
    select_pseudo_label = []
    indices_list = []
    data_after = []
    for i in range(class_num):
        # logits calculate KL divergence
        _, index_i = logits[:, i:i+1].squeeze().sort(descending=True)
        logits_i = logits[index_i[0:after_shots*2]]
        kl_i_to_i = torch.zeros(len(logits_i))
        for j in range(len(logits_i)):
            kl_i_to_i[j] = torch.sum(logits_i[j] * torch.log(logits_i[j] / logits_i))
        _, index = kl_i_to_i.sort()
        indices = index_i[index[0:after_shots]]
        indices_list.append(indices)
        cls_tokens_after.append(cls_tokens[indices])
        label = torch.zeros(min(len(indices), after_shots), dtype=torch.int64).to(torch.device("cuda:0")) + i
        label_after.append(label)
        if not cache:
            select_pseudo_label.append(pseudo_label[indices])
        true_label.append(test_label[indices])
    cls_tokens_after = torch.cat(cls_tokens_after, dim=0)
    label_after = torch.cat(label_after, dim=0)
    if not cache:
        select_pseudo_label = torch.cat(select_pseudo_label, dim=0)
    true_label = torch.cat(true_label, dim=0)
    indices_list = torch.cat(indices_list, dim=0)
    num = 0
    for i in indices_list:
        if cache:
            data_after.append(data[i])
            if label_after[num] != true_label[num]:
                item = Datum(
                    impath=data_after[num].impath,
                    label=int(label_after[num]),
                    domain=data_after[num].domain,
                    classname=data_after[num].classname
                )
                data_after[num] = item
            num = num + 1
        else:
            data_after.append(data[i])
            if select_pseudo_label[num] != true_label[num]:
                item = Datum(
                    impath=data_after[num].impath,
                    label=int(select_pseudo_label[num]),
                    domain=data_after[num].domain,
                    classname=data_after[num].classname
                )
                data_after[num] = item
            num = num + 1
    correct = (true_label == label_after).sum()
    print("correct num:", correct, "acc:", correct / (after_shots * class_num))
    if not cache:
        correct1 = (true_label == select_pseudo_label).sum()
        print("correct1 num:", correct1, "acc:", correct1 / (after_shots * class_num))
    if cache:
        torch.save(cls_tokens_after, cache_dir+'/cache_keys.pt')
        torch.save(label_after, cache_dir+'/cache_values.pt')
    return data_after


def get_sim_based_kl(new_cache_keys, new_test_feature, new_clip_weights):
    new_cache_logits = F.softmax(10. * new_cache_keys @ new_clip_weights, dim=1)
    new_test_logits = F.softmax(10. * new_test_feature @ new_clip_weights, dim=1)
    kl = torch.empty(len(new_test_logits), len(new_cache_logits), dtype=torch.float16).to(torch.device("cuda:0"))
    for i in range(len(new_test_logits)):
        kl[i] = (new_test_logits[i] * torch.log(new_test_logits[i] / new_cache_logits)).sum(dim=1)
    kl_mean = kl.mean(dim=1).unsqueeze(1)
    kl_std = kl.std(dim=1).unsqueeze(1)
    kl_norm = (kl - kl_mean) / kl_std
    kl_norm_max = kl_norm.max(dim=1).values.unsqueeze(1)
    kl_norm_min = kl_norm.min(dim=1).values.unsqueeze(1)
    sim = 2 * (1 - (kl_norm - kl_norm_min) / (kl_norm_max - kl_norm_min))
    return sim


def get_clip_weights(cfg, clip_model, classnames):
    with torch.no_grad():
        if cfg.DATASET.NAME == 'VisDA17':   # ZSCLIP
            temp = CUSTOM_TEMPLATES[cfg.DATASET.NAME]
            prompts = [temp.format(c) for c in classnames]
            prompts = torch.cat([clip.tokenize(p) for p in prompts])
            prompts = prompts.to(torch.device("cuda"))
            clip_weights = clip_model.encode_text(prompts)
        else:   # GPT prompt ZSCLIP
            PATH_TO_PROMPTS = cfg.TRAINER.TFUP.PATH_TO_PROMPTS
            with open(PATH_TO_PROMPTS) as f:
                gpt3_prompts = json.load(f)
            text_features = []
            i = 0
            for classname in tqdm(classnames):
                texts = []
                for t in gpt3_prompts[classnames[i]]:
                    texts.append(t)
                texts = clip.tokenize(texts, truncate=True).cuda()  # tokenize
                class_embeddings = clip_model.encode_text(texts)  # embed with text encoder
                class_embedding = class_embeddings.mean(dim=0)
                text_features.append(class_embedding)
                i += 1
            clip_weights = torch.stack(text_features, dim=0).cuda()

    return clip_weights


def save_clip_image_features(cfg, clip_model, loader, split='test'):
    cache_keys = []
    cache_values = []

    with torch.no_grad():
        train_features = []
        for i, batch in enumerate(tqdm(loader)):
            images, target = batch["img"], batch["label"]
            images, target = images.cuda(), target.cuda()
            image_features = clip_model.encode_image(images)
            train_features.append(image_features)
            cache_values.append(target)
        cache_keys.append(torch.cat(train_features, dim=0).unsqueeze(0))

    cache_keys = torch.cat(cache_keys, dim=0).mean(dim=0)
    cache_values = torch.cat(cache_values, dim=0)

    cache_dir = 'cache_dir/' + cfg.DATASET.NAME + '/' + cfg.DATASET.TARGET_DOMAINS
    torch.save(cache_keys, cache_dir + '/'+split+'_keys' + ".pt")
    torch.save(cache_values, cache_dir + '/'+split+'_values' + ".pt")
