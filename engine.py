import torch
from torchvision.transforms.functional import resize
from torchvision.io import read_image

from fastai.vision.all import *
from fastai.data.all import *

import glob

#Import libraries
import os
import concurrent.futures
from GoogleImageScraper import GoogleImageScraper
from patch import webdriver_executable

### Configuration ###
#Add new search key into array ["cat","t-shirt","apple","orange","pear","fish"]
search_keys = list(set(["washing machine", "coffee machines"]))
number_of_images = 10                # Desired number of images
model = resnet34(pretrained=True) # Desired pretrained model
n_epoch = 3 # Desired number of epochs for fine-tuning 

#Define file path
webdriver_path = os.path.normpath(os.path.join(os.getcwd(), 'webdriver', webdriver_executable()))
image_path = os.path.normpath(os.path.join(os.getcwd(), 'images'))

#Parameters
headless = True                     # True = No Chrome GUI
min_resolution = (0, 0)             # Minimum desired image resolution
max_resolution = (9999, 9999)       # Maximum desired image resolution
max_missed = 10                     # Max number of failed images before exit
number_of_workers = 1               # Number of "workers" used
keep_filenames = False              # Keep original URL image filenames

def worker_thread(search_key):
    image_scraper = GoogleImageScraper(
        webdriver_path, 
        image_path, 
        search_key, 
        number_of_images, 
        headless, 
        min_resolution, 
        max_resolution, 
        max_missed)
    image_urls = image_scraper.find_image_urls()
    image_scraper.save_images(image_urls, keep_filenames)

    #Release resources
    del image_scraper

#Run each search_key in a separate thread
#Automatically waits for all threads to finish
#Removes duplicate strings from search_keys
with concurrent.futures.ThreadPoolExecutor(max_workers=number_of_workers) as executor:
    executor.map(worker_thread, search_keys)


def min_max_norm(mon_image):
    image_normalized = mon_image.type(torch.FloatTensor) # instantier le Tensor normalisé
    for i in range(3):
        image_normalized[i] = (mon_image[i] - mon_image[i].min())/(mon_image[i].max() - mon_image[i].min())
    return image_normalized

# On définit un nouveau dataloader
def my_dataloader(image_list, category_list):
    def pass_indices(indices):    # par défaut, "dblock.dataloaders" prend une liste d'indices en argument, il faut définir comment les gérer
        return indices

    def get_x(idx):         # comment charger les inputs
#         im = image_list[idx]
        im = resize(image_list[idx], [224,224])
        return im

    def get_y(idx):         # comment charger les outputs
        return category_list[idx]

    # on crée le DataBlock (note: pas besoin de splitter parce qu'on a que 2 images donc pas assez pour entrainement/validation)
    dblock = DataBlock( blocks=(TransformBlock, CategoryBlock),
                        get_items=pass_indices,
                        get_x=get_x,
                        get_y=get_y,
#                         item_tfms=[Resize(512)],
                        batch_tfms = Normalize.from_stats(*imagenet_stats)
                      )

    nombre_images = len(image_list) #la première dimension du tensor d'entrée nous donne le nombre d'éléments à traiter
    dls = dblock.dataloaders(list(range(nombre_images)), bs=2)

    return dls

# Récupération des paths
coffee_machines = glob.glob("images/coffee_machine/*.jpeg")
washing_machines = glob.glob("images/washing_machine/*.jpeg")

# On place toutes les images (matrices 3D) dans une liste
im_list = [] # créer une liste vide
cat = []
for i,subdir in enumerate(search_keys):
	image_paths = glob.glob(f'images/{subdir}/*.jpeg')
	for _,item in enumerate(image_paths):
	    im = read_image(item)
	    im = min_max_norm(im)
	    im_list.append(im)
	    cat.append(i)

dls = my_dataloader(im_list,cat)

# configure the fine tuning
# define the layers to freeze
for name, param in model.named_parameters():
    if name.startswith('layer4'):
        param.requires_grad = True
    else:
        param.requires_grad = False
        

learn = Learner(dls, model)
learn.fit(n_epoch, lr=1e-3)

results = learn.get_preds(with_decoded=True, with_input=True)

for i,image in enumerate(results[0]):
    plt.imshow(image.permute(1,2,0).cpu())
    plt.savefig(f'results/label_{results[3][i]}.png', bbox_inches='tight')