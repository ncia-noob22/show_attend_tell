from torchvision.datasets import Flickr30k
import torch.utils.data as data
import torchvision.transforms as transforms

# !
dir_img = "/data/Flickr30k/img"
path_ann = "/data/Flickr30k/ann/results_20130124.token"
num_batch = 1
# !


def get_dataloader(dir_img, path_ann, num_batch, **kwargs):
    transform = transforms.Compose([transforms.Normalize(), transforms.ToTensor()])

    dataset = Flickr30k(dir_img, path_ann, transform=transform)
    dataloader = data.DataLoader(dataset, batch_size=num_batch, shuffle=True)
    return dataloader


# #!
# ann = """
# 4982521008.jpg#0	The man , only visible as a black silhouette , took a picture along a brightly colored wall full of graffiti .
# 4982521008.jpg#1	A photographer stands in the shadows in front of a wall covered with graffiti .
# 4982521008.jpg#2	A man holding a camera in front of a graffiti covered wall .
# 4982521008.jpg#3	A man taking pictures in the shadows .
# 4982521008.jpg#4	A photographer taking pictures .
# """

# data = tensor(
#     [
#         [
#             [
#                 [0.3098, 0.2863, 0.2902, ..., 0.7765, 0.6667, 0.6196],
#                 [0.7137, 0.7255, 0.7529, ..., 0.6275, 0.5176, 0.4706],
#                 [0.9804, 0.9882, 0.9922, ..., 0.5412, 0.4157, 0.3765],
#                 ...,
#                 [0.1216, 0.1529, 0.1529, ..., 0.1098, 0.1216, 0.1294],
#                 [0.1686, 0.1294, 0.1255, ..., 0.0824, 0.0863, 0.0941],
#                 [0.1686, 0.1608, 0.1373, ..., 0.0980, 0.0941, 0.1059],
#             ],
#             [
#                 [0.3725, 0.3569, 0.3765, ..., 0.9882, 0.9961, 0.9961],
#                 [0.7137, 0.7137, 0.7333, ..., 0.9922, 0.9843, 0.9804],
#                 [0.9961, 0.9922, 0.9922, ..., 0.9882, 0.9882, 0.9882],
#                 ...,
#                 [0.1059, 0.1373, 0.1333, ..., 0.1098, 0.1216, 0.1294],
#                 [0.1569, 0.1176, 0.1098, ..., 0.0745, 0.0824, 0.0902],
#                 [0.1569, 0.1490, 0.1216, ..., 0.0784, 0.0784, 0.0863],
#             ],
#             [
#                 [0.4314, 0.4157, 0.4235, ..., 0.2627, 0.2706, 0.2745],
#                 [0.7176, 0.7255, 0.7451, ..., 0.2157, 0.2902, 0.3176],
#                 [0.9843, 0.9843, 0.9882, ..., 0.2667, 0.2902, 0.2863],
#                 ...,
#                 [0.0980, 0.1255, 0.1255, ..., 0.1098, 0.1216, 0.1294],
#                 [0.1373, 0.0980, 0.0902, ..., 0.0784, 0.0824, 0.0902],
#                 [0.1373, 0.1294, 0.1059, ..., 0.0941, 0.0902, 0.1020],
#             ],
#         ]
#     ]
# )
# labels = [
#     (
#         "The man , only visible as a black silhouette , took a picture along a brightly colored wall full of graffiti .",
#     ),
#     (
#         "A photographer stands in the shadows in front of a wall covered with graffiti .",
#     ),
#     ("A man holding a camera in front of a graffiti covered wall .",),
#     ("A man taking pictures in the shadows .",),
#     ("A photographer taking pictures .",),
# ]
# #!
