import argparse

import torch

from clip.clip import _MODELS, _download
from clip.model import build_model
from read_data import *
from utils import *
from dassl.config import get_cfg_default
from dassl.utils import set_random_seed


def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME  # cfg.MODEL.BACKBONE.NAME
    url = _MODELS[backbone_name]
    model_path = _download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = build_model(state_dict or model.state_dict())

    return model


def extend_cfg(cfg):
    """
    Add new config variables.

    E.g.
        from yacs.config import CfgNode as CN
        cfg.TRAINER.MY_MODEL = CN()
        cfg.TRAINER.MY_MODEL.PARAM_A = 1.
        cfg.TRAINER.MY_MODEL.PARAM_B = 0.5
        cfg.TRAINER.MY_MODEL.PARAM_C = False
    """
    from yacs.config import CfgNode as CN

    cfg.TRAINER.TFUP = CN()
    cfg.TRAINER.TFUP.beta = 1
    cfg.TRAINER.TFUP.alpha = 0.3
    cfg.TRAINER.TFUP.ratio_beta = 0.5
    cfg.TRAINER.TFUP.ratio_alpha = 0.2
    cfg.TRAINER.TFUP.cache_proportion = 0.3
    cfg.TRAINER.TFUP.training_proportion = 0.75
    cfg.TRAINER.TFUP.PATH_TO_PROMPTS = './gpt3_prompts/prompts/gpt-3.5-turbo-instruct_office_home.json'


def get_arguments():

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', dest='config', help='settings of TFUP in yaml format')
    parser.add_argument("--domain_name", dest='domain_name', type=str, default="", help="domain name")
    parser.add_argument("--eval-only", action="store_true", help="evaluation only")
    parser.add_argument("--training", action="store_true", help="training")
    args = parser.parse_args()
    return args


def inference_TFUP_T(cfg, cache_keys, cache_values, test_features, test_labels, clip_weights, cache_dir):
    ratio_alpha, beta = cfg.TRAINER.TFUP.ratio_alpha, cfg.TRAINER.TFUP.beta
    clip_weights = clip_weights.permute(1, 0)

    text_adapter = torch.load(cache_dir + "/text_adapter.pt")
    image_adapter = torch.load(cache_dir + "/image_adapter.pt")
    text_adapter.eval()
    image_adapter.eval()
    with torch.no_grad():
        new_cache_keys, new_clip_weights, new_cache_values = text_adapter(cache_keys, clip_weights, cache_values)
        x = image_adapter(test_features)
        test_feature1 = ratio_alpha * x + (1 - ratio_alpha) * test_features
        test_feature1 = test_feature1 / test_feature1.norm(dim=-1, keepdim=True)
        logits = 100. * test_feature1 @ new_clip_weights
    acc = cls_acc(logits, test_labels)
    print("\n**** TFUP-T's inference accuracy: {:.2f}. ****\n".format(acc))


def TFUP(cfg, cache_keys, cache_values, test_features, test_labels, clip_weights):
    beta, alpha = cfg.TRAINER.TFUP.beta, cfg.TRAINER.TFUP.alpha
    clip_weights = clip_weights.permute(1, 0)

    clip_weights = clip_weights / clip_weights.norm(dim=0, keepdim=True)
    test_features = test_features / test_features.norm(dim=-1, keepdim=True)
    cache_keys = cache_keys / cache_keys.norm(dim=-1, keepdim=True)

    R_fW = 100. * test_features @ clip_weights
    R_fF = test_features @ cache_keys.t()
    sim = get_sim_based_kl(cache_keys, test_features, clip_weights)
    R_fF = R_fF * sim
    cache_logits = ((-1) * (beta - beta * R_fF)).exp() @ cache_values
    ape_logits = R_fW + cache_logits * alpha
    acc1 = cls_acc(R_fW, test_labels)
    acc2 = cls_acc(ape_logits, test_labels)
    print("\n**** Zero-shot CLIP's test accuracy: {:.2f}. ****\n".format(acc1))
    print("\n**** Our training free accuracy: {:.2f}. ****\n".format(acc2))


def TFUP_T(cfg, cache_keys, cache_values, test_features, test_labels, clip_weights, clip_model, refine_loader,
           total_loader, save_path, train_features=None):
    beta, alpha = cfg.TRAINER.TFUP.beta, cfg.TRAINER.TFUP.alpha
    ratio_beta, ratio_alpha = cfg.TRAINER.TFUP.ratio_beta, cfg.TRAINER.TFUP.ratio_alpha

    clip_weights = clip_weights.permute(1, 0)

    text_adapter = TextAdapter(clip_weights, ratio_beta).cuda()
    image_adapter = Adapter(512, 4).to(torch.float16).cuda()
    optimizer = torch.optim.AdamW([
        {'params': text_adapter.parameters()},
        {'params': image_adapter.parameters()}
    ], lr=cfg.OPTIM.LR, eps=0.003, weight_decay=1e-1)
    warmup_epoch, main_epoch = cfg.OPTIM.WARMUP_EPOCH, cfg.OPTIM.MAX_EPOCH - cfg.OPTIM.WARMUP_EPOCH
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg.OPTIM.MAX_EPOCH * len(refine_loader))
    Loss = SmoothCrossEntropy()
    best_acc, best_epoch = 0.0, 0

    for train_idx in range(cfg.OPTIM.MAX_EPOCH):
        # Train
        text_adapter.train()
        image_adapter.train()
        correct_samples, all_samples = 0, 0
        loss_list = []
        loss, ce_loss, mg_loss = 0.0, 0.0, 0.0
        print('Train Epoch: {:} / {:}'.format(train_idx, cfg.OPTIM.MAX_EPOCH))
        if train_idx < warmup_epoch:
            for i, batch in enumerate(tqdm(refine_loader)):
                images, target = batch["img"], batch["label"]
                images, target = images.cuda(), target.cuda()
                with torch.no_grad():
                    image_features = clip_model.encode_image(images)
                x = image_adapter(image_features)
                image_features = ratio_alpha * x + (1 - ratio_alpha) * image_features
                image_features = image_features/image_features.norm(dim=-1, keepdim=True)
                new_cache_keys, new_clip_weights, _ = text_adapter(cache_keys, clip_weights, cache_values)
                logits = 100. * image_features @ new_clip_weights

                loss = Loss(logits, target)

                acc = cls_acc(logits, target)
                correct_samples += acc / 100 * len(logits)
                all_samples += len(logits)
                loss_list.append(loss.item())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
        else:
            for i, batch in enumerate(tqdm(total_loader)):
                images, target, idx = batch["img"], batch["label"], batch["index"]
                images, target = images.cuda(), target.cuda()
                with torch.no_grad():
                    image_features = clip_model.encode_image(images)
                if cfg.DATASET.NAME == 'DomainNet':
                    image_features_weak = train_features[idx]
                else:
                    image_features_weak = test_features[idx]
                x = image_adapter(image_features)
                x_weak = image_adapter(image_features_weak)
                image_features = ratio_alpha * x + (1 - ratio_alpha) * image_features
                image_features_weak = ratio_alpha * x_weak + (1 - ratio_alpha) * image_features_weak
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                image_features_weak = image_features_weak / image_features_weak.norm(dim=-1, keepdim=True)
                new_cache_keys, new_clip_weights, new_cache_values = text_adapter(cache_keys, clip_weights, cache_values)

                sim = get_sim_based_kl(new_cache_keys, image_features_weak, new_clip_weights)
                R_fF = image_features_weak @ new_cache_keys.half().t()
                R_fF = R_fF * sim
                cache_logits = ((-1) * (beta - beta * R_fF)).exp() @ new_cache_values
                R_fW = 100. * image_features_weak @ new_clip_weights
                strong_logits = 100. * image_features @ new_clip_weights
                weak_logits = R_fW + cache_logits * alpha
                confident_idx = F.softmax(weak_logits).max(dim=1).values > 0.9
                confident_pseudo_label = (weak_logits.max(dim=1).indices)[confident_idx]
                true_label = target[confident_idx]
                confident_strong_logits = strong_logits[confident_idx]
                correct_num = (true_label == confident_pseudo_label).sum()
                # print('confident_num: {:.4f}'.format(len(confident_pseudo_label)))
                # print('correct_num: {:.4f}'.format(correct_num))
                if len(confident_pseudo_label) > 0:
                    ce_loss = Loss(confident_strong_logits, confident_pseudo_label)
                else:
                    ce_loss = 0.0
                mg_loss = marginal_loss(strong_logits)
                loss = ce_loss + 0.3 * mg_loss

                correct_samples += correct_num
                all_samples += len(confident_pseudo_label)
                loss_list.append(loss.item())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

        current_lr = scheduler.get_last_lr()[0]
        print('LR: {:.6f}, Acc: {:.4f} ({:}/{:}), Loss: {:.4f}'.format(current_lr, correct_samples / all_samples,
                                                                       correct_samples, all_samples,
                                                                       sum(loss_list) / len(loss_list)))
        # Eval
        text_adapter.eval()
        image_adapter.eval()
        with torch.no_grad():
            new_cache_keys, new_clip_weights, new_cache_values = text_adapter(cache_keys, clip_weights, cache_values)
            x = image_adapter(test_features)
            test_feature1 = ratio_alpha * x + (1 - ratio_alpha) * test_features
            test_feature1 = test_feature1 / test_feature1.norm(dim=-1, keepdim=True)
            sim = get_sim_based_kl(new_cache_keys, test_feature1, new_clip_weights)
            R_fF = test_feature1 @ new_cache_keys.half().t()
            R_fF = R_fF * sim
            cache_logits = ((-1) * (beta - beta * R_fF)).exp() @ new_cache_values
            logits1 = 100. * test_feature1 @ new_clip_weights
            logits2 = logits1 + cache_logits * alpha
        acc = cls_acc(logits1, test_labels)
        acc1 = cls_acc(logits2, test_labels)
        print("**** TFUP-T's test accuracy: {:.2f}. ****\n".format(acc))
        print("**** TFUP-T's test accuracy1: {:.2f}. ****\n".format(acc1))
        if acc > best_acc:
            best_acc = acc
            best_epoch = train_idx
            torch.save(text_adapter, save_path + "/text_adapter.pt")
            torch.save(image_adapter, save_path + "/image_adapter.pt")
    print(f"**** After fine-tuning, TFUP-T's best test accuracy: {best_acc:.2f}, at epoch: {best_epoch}. ****\n")


def main():
    args = get_arguments()
    cfg = get_cfg_default()
    extend_cfg(cfg)
    cfg.merge_from_file(args.config)
    cfg.DATASET.TARGET_DOMAINS = args.domain_name

    if cfg.SEED >= 0:
        print("Setting fixed seed: {}".format(cfg.SEED))
        set_random_seed(cfg.SEED)

    device = torch.device("cuda")
    cache_proportion = cfg.TRAINER.TFUP.cache_proportion
    training_proportion = cfg.TRAINER.TFUP.training_proportion
    if cfg.DATASET.NAME == 'Office31' and cfg.DATASET.TARGET_DOMAINS == 'amazon':
        training_proportion = 0.75

    clip_model = load_clip_to_cpu(cfg)
    clip_model.to(device)

    total_data, classnames = read_data(cfg)
    classnames = [c.replace("_", " ") for c in classnames]

    # save clip features and labels
    if cfg.DATASET.NAME == 'DomainNet':     # test split and train split
        domainnet_train_loader = build_dataloader(cfg, total_data, is_train=False)
        input_domains = []
        input_domains.append(cfg.DATASET.TARGET_DOMAINS)
        test_data, _ = read_data_domainnet(input_domains, cfg.DATASET.ROOT, split='test')
        domainnet_test_loader = build_dataloader(cfg, test_data, is_train=False)
        save_clip_image_features(cfg, clip_model, domainnet_test_loader)
        save_clip_image_features(cfg, clip_model, domainnet_train_loader, split='train')
    else:
        test_loader = build_dataloader(cfg, total_data, is_train=False)
        save_clip_image_features(cfg, clip_model, test_loader)

    cache_dir = 'cache_dir/'+cfg.DATASET.NAME+'/'+cfg.DATASET.TARGET_DOMAINS

    test_features = torch.load(cache_dir+'/test_keys.pt')
    test_labels = torch.load(cache_dir+'/test_values.pt')
    if cfg.DATASET.NAME == 'DomainNet':
        train_features = torch.load(cache_dir + '/train_keys.pt')
        train_labels = torch.load(cache_dir + '/train_values.pt')

    cache_shots = int(len(total_data) * cache_proportion / len(classnames))
    training_shots = int(len(total_data) * training_proportion / len(classnames))
    cache_shots = max(9, cache_shots)
    cache_shots = min(16, cache_shots)
    training_shots = max(9, training_shots)
    print("cache_shots:{}, training_shots:{}".format(cache_shots, training_shots))

    clip_weights = get_clip_weights(cfg, clip_model, classnames)

    if cfg.DATASET.NAME == 'DomainNet':
        refine_few_shots_without_label(classnames, clip_weights, train_features, train_labels, total_data, cache_shots,
                                       cache_dir)
        refine_data = refine_few_shots_without_label(classnames, clip_weights, train_features, train_labels, total_data,
                                                     training_shots, cache_dir, cache=False)
    else:
        refine_few_shots_without_label(classnames, clip_weights, test_features, test_labels, total_data, cache_shots,
                                       cache_dir)
        refine_data = refine_few_shots_without_label(classnames, clip_weights, test_features, test_labels, total_data,
                                                     training_shots, cache_dir, cache=False)

    total_loader = build_dataloader(cfg, total_data, is_train=True)
    refine_loader = build_dataloader(cfg, refine_data, is_train=True)

    cache_keys = torch.load(cache_dir+'/cache_keys.pt')
    cache_values = torch.load(cache_dir+'/cache_values.pt')
    cache_values = cache_values.long()
    cache_values = F.one_hot(cache_values).half()

    # ------------------------------------------  TFUP  ------------------------------------------
    TFUP(cfg, cache_keys, cache_values, test_features, test_labels, clip_weights)

    # ------------------------------------------ TFUP-T ------------------------------------------
    if args.eval_only:
        inference_TFUP_T(cfg, cache_keys, cache_values, test_features, test_labels, clip_weights, cache_dir)
        return
    if args.training:
        if cfg.DATASET.NAME == 'DomainNet':
            TFUP_T(cfg, cache_keys, cache_values, test_features, test_labels, clip_weights, clip_model, refine_loader,
                   total_loader, cache_dir, train_features)
        else:
            TFUP_T(cfg, cache_keys, cache_values, test_features, test_labels, clip_weights, clip_model, refine_loader,
                   total_loader, cache_dir)


if __name__ == '__main__':
    main()
