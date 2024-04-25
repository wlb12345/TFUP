# How to Install Datasets

`$DATA` denotes the location where datasets are installed, e.g.

```
$DATA/
|–– office31/
|–– office_home/
|–– visda17/
|–– domainnet/
```

[Datasets list:](#Datasets list)
- [Office-31](#office-31)
- [Office-Home](#office-home)
- [VisDA17](#visda17)
- [DomainNet](#domainnet)

## Datasets list

### Office-31

Download link: https://people.eecs.berkeley.edu/~jhoffman/domainadapt/#datasets_code.

File structure:

```
office31/
|–– amazon/
|   |–– back_pack/
|   |–– bike/
|   |–– ...
|–– dslr/
|   |–– back_pack/
|   |–– bike/
|   |–– ...
|–– webcam/
|   |–– back_pack/
|   |–– bike/
|   |–– ...
```

Note that within each domain folder you need to move all class folders out of the `images/` folder and then delete the `images/` folder.

### Office-Home

Download link: http://hemanthdv.org/OfficeHome-Dataset/.

File structure:

```
office_home/
|–– art/
|–– clipart/
|–– product/
|–– real_world/
```

### VisDA17

Download link: http://ai.bu.edu/visda-2017/.

Once the download is finished, the file structure will look like

```
visda17/
|–– train/
|–– test/
|–– validation/
```

### DomainNet

Download link: http://ai.bu.edu/M3SDA/. (Please download the cleaned version of split files)

File structure:

```
domainnet/
|–– clipart/
|–– infograph/
|–– painting/
|–– quickdraw/
|–– real/
|–– sketch/
|–– splits/
|   |–– clipart_train.txt
|   |–– clipart_test.txt
|   |–– ...
```
