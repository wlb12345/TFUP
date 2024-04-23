import os.path as osp
from dassl.data.datasets import Datum
from dassl.data.data_manager import build_data_loader
from dassl.data.transforms import build_transform
from dassl.utils import listdir_nohidden


def build_dataloader(cfg, dataset, is_train):
    if is_train:
        tfm_train = build_transform(cfg, is_train=True)
        data_loader = build_data_loader(
            cfg,
            sampler_type=cfg.DATALOADER.TRAIN_X.SAMPLER,
            data_source=dataset,
            batch_size=cfg.DATALOADER.TRAIN_X.BATCH_SIZE,
            n_domain=cfg.DATALOADER.TRAIN_X.N_DOMAIN,
            n_ins=cfg.DATALOADER.TRAIN_X.N_INS,
            tfm=tfm_train,
            is_train=True,
            dataset_wrapper=None
        )
    else:
        tfm_test = build_transform(cfg, is_train=False)
        data_loader = build_data_loader(
            cfg,
            sampler_type=cfg.DATALOADER.TEST.SAMPLER,
            data_source=dataset,
            batch_size=cfg.DATALOADER.TEST.BATCH_SIZE,
            tfm=tfm_test,
            is_train=False,
            dataset_wrapper=None
        )
    return data_loader


def read_data_visda17(dname, root_path):
    dname = dname[0]
    filedir = "train" if dname == "synthetic" else "validation"
    dataset_dir = root_path     # "D:/datasets/visda17"
    image_list = osp.join(dataset_dir, filedir, "image_list.txt")
    items = []
    classnames = ['aeroplane', 'bicycle', 'bus', 'car', 'horse', 'knife', 'motorcycle', 'person', 'plant', 'skateboard',
                  'train', 'truck']
    # There is only one source domain
    domain = 0

    with open(image_list, "r") as f:
        lines = f.readlines()

        for line in lines:
            line = line.strip()
            impath, label = line.split(" ")
            classname = impath.split("/")[0]
            impath = osp.join(dataset_dir, filedir, impath)
            label = int(label)
            item = Datum(
                impath=impath,
                label=label,
                domain=domain,
                classname=classname
            )
            items.append(item)

    return items, classnames

def read_data_officehome(input_domains, root_path):
    items = []
    classnames = []
    dataset_dir = root_path     # "D:/datasets/office_home"
    for domain, dname in enumerate(input_domains):
        domain_dir = osp.join(dataset_dir, dname)
        class_names = listdir_nohidden(domain_dir)
        class_names.sort()

        for label, class_name in enumerate(class_names):
            class_path = osp.join(domain_dir, class_name)
            imnames = listdir_nohidden(class_path)

            for imname in imnames:
                impath = osp.join(class_path, imname)
                item = Datum(
                    impath=impath,
                    label=label,
                    domain=domain,
                    classname=class_name.lower(),
                )
                items.append(item)
            classnames.append(class_name.lower())

    return items, classnames


def read_data_office31(input_domains, root_path):
    items = []
    classnames = []
    dataset_dir = root_path     # "D:/datasets/office31"
    for domain, dname in enumerate(input_domains):
        domain_dir = osp.join(dataset_dir, dname)
        class_names = listdir_nohidden(domain_dir)
        class_names.sort()

        for label, class_name in enumerate(class_names):
            class_path = osp.join(domain_dir, class_name)
            imnames = listdir_nohidden(class_path)

            for imname in imnames:
                impath = osp.join(class_path, imname)
                item = Datum(
                    impath=impath,
                    label=label,
                    domain=domain,
                    classname=class_name
                )
                items.append(item)
            classnames.append(class_name.lower())
    return items, classnames


def read_data_domainnet(input_domains, root_path, split="train"):
    items = []
    classnames = ['aircraft_carrier', 'airplane', 'alarm_clock', 'ambulance', 'angel', 'animal_migration', 'ant',
                  'anvil', 'apple', 'arm', 'asparagus', 'axe', 'backpack', 'banana', 'bandage', 'barn', 'baseball',
                  'baseball_bat', 'basket', 'basketball', 'bat', 'bathtub', 'beach', 'bear', 'beard', 'bed', 'bee',
                  'belt', 'bench', 'bicycle', 'binoculars', 'bird', 'birthday_cake', 'blackberry', 'blueberry', 'book',
                  'boomerang', 'bottlecap', 'bowtie', 'bracelet', 'brain', 'bread', 'bridge', 'broccoli', 'broom',
                  'bucket', 'bulldozer', 'bus', 'bush', 'butterfly', 'cactus', 'cake', 'calculator', 'calendar',
                  'camel', 'camera', 'camouflage', 'campfire', 'candle', 'cannon', 'canoe', 'car', 'carrot', 'castle',
                  'cat', 'ceiling_fan', 'cello', 'cell_phone', 'chair', 'chandelier', 'church', 'circle', 'clarinet',
                  'clock', 'cloud', 'coffee_cup', 'compass', 'computer', 'cookie', 'cooler', 'couch', 'cow', 'crab',
                  'crayon', 'crocodile', 'crown', 'cruise_ship', 'cup', 'diamond', 'dishwasher', 'diving_board', 'dog',
                  'dolphin', 'donut', 'door', 'dragon', 'dresser', 'drill', 'drums', 'duck', 'dumbbell', 'ear', 'elbow',
                  'elephant', 'envelope', 'eraser', 'eye', 'eyeglasses', 'face', 'fan', 'feather', 'fence', 'finger',
                  'fire_hydrant', 'fireplace', 'firetruck', 'fish', 'flamingo', 'flashlight', 'flip_flops',
                  'floor_lamp', 'flower', 'flying_saucer', 'foot', 'fork', 'frog', 'frying_pan', 'garden',
                  'garden_hose', 'giraffe', 'goatee', 'golf_club', 'grapes', 'grass', 'guitar', 'hamburger', 'hammer',
                  'hand', 'harp', 'hat', 'headphones', 'hedgehog', 'helicopter', 'helmet', 'hexagon', 'hockey_puck',
                  'hockey_stick', 'horse', 'hospital', 'hot_air_balloon', 'hot_dog', 'hot_tub', 'hourglass', 'house',
                  'house_plant', 'hurricane', 'ice_cream', 'jacket', 'jail', 'kangaroo', 'key', 'keyboard', 'knee',
                  'knife', 'ladder', 'lantern', 'laptop', 'leaf', 'leg', 'light_bulb', 'lighter', 'lighthouse',
                  'lightning', 'line', 'lion', 'lipstick', 'lobster', 'lollipop', 'mailbox', 'map', 'marker', 'matches',
                  'megaphone', 'mermaid', 'microphone', 'microwave', 'monkey', 'moon', 'mosquito', 'motorbike',
                  'mountain', 'mouse', 'moustache', 'mouth', 'mug', 'mushroom', 'nail', 'necklace', 'nose', 'ocean',
                  'octagon', 'octopus', 'onion', 'oven', 'owl', 'paintbrush', 'paint_can', 'palm_tree', 'panda',
                  'pants', 'paper_clip', 'parachute', 'parrot', 'passport', 'peanut', 'pear', 'peas', 'pencil',
                  'penguin', 'piano', 'pickup_truck', 'picture_frame', 'pig', 'pillow', 'pineapple', 'pizza', 'pliers',
                  'police_car', 'pond', 'pool', 'popsicle', 'postcard', 'potato', 'power_outlet', 'purse', 'rabbit',
                  'raccoon', 'radio', 'rain', 'rainbow', 'rake', 'remote_control', 'rhinoceros', 'rifle', 'river',
                  'roller_coaster', 'rollerskates', 'sailboat', 'sandwich', 'saw', 'saxophone', 'school_bus',
                  'scissors', 'scorpion', 'screwdriver', 'sea_turtle', 'see_saw', 'shark', 'sheep', 'shoe', 'shorts',
                  'shovel', 'sink', 'skateboard', 'skull', 'skyscraper', 'sleeping_bag', 'smiley_face', 'snail',
                  'snake', 'snorkel', 'snowflake', 'snowman', 'soccer_ball', 'sock', 'speedboat', 'spider', 'spoon',
                  'spreadsheet', 'square', 'squiggle', 'squirrel', 'stairs', 'star', 'steak', 'stereo', 'stethoscope',
                  'stitches', 'stop_sign', 'stove', 'strawberry', 'streetlight', 'string_bean', 'submarine', 'suitcase',
                  'sun', 'swan', 'sweater', 'swing_set', 'sword', 'syringe', 'table', 'teapot', 'teddy-bear',
                  'telephone', 'television', 'tennis_racquet', 'tent', 'The_Eiffel_Tower', 'The_Great_Wall_of_China',
                  'The_Mona_Lisa', 'tiger', 'toaster', 'toe', 'toilet', 'tooth', 'toothbrush', 'toothpaste', 'tornado',
                  'tractor', 'traffic_light', 'train', 'tree', 'triangle', 'trombone', 'truck', 'trumpet', 't-shirt',
                  'umbrella', 'underwear', 'van', 'vase', 'violin', 'washing_machine', 'watermelon', 'waterslide',
                  'whale', 'wheel', 'windmill', 'wine_bottle', 'wine_glass', 'wristwatch', 'yoga', 'zebra', 'zigzag']
    split_dir = root_path+'/splits'     # "/media/SN570/Datasets/domainnet/splits"
    dataset_dir = root_path     # "/media/SN570/Datasets/domainnet"
    for domain, dname in enumerate(input_domains):
        filename = dname + "_" + split + ".txt"
        split_file = osp.join(split_dir, filename)

        with open(split_file, "r") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                impath, label = line.split(" ")
                classname = impath.split("/")[1]
                impath = osp.join(dataset_dir, impath)
                label = int(label)
                item = Datum(
                    impath=impath,
                    label=label,
                    domain=domain,
                    classname=classname
                )
                items.append(item)

    return items, classnames


def read_data(cfg):
    input_domains = []
    input_domains.append(cfg.DATASET.TARGET_DOMAINS)
    root_path = cfg.DATASET.ROOT
    if cfg.DATASET.NAME == 'OfficeHome':
        total_data, classnames = read_data_officehome(input_domains, root_path)
    elif cfg.DATASET.NAME == 'Office31':
        total_data, classnames = read_data_office31(input_domains, root_path)
    elif cfg.DATASET.NAME == 'VisDA17':
        total_data, classnames = read_data_visda17(input_domains, root_path)
    else:
        total_data, classnames = read_data_domainnet(input_domains, root_path)
    return total_data, classnames
