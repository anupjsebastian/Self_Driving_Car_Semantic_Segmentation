import os
from PIL import Image , ImageOps, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True



def resize_image(dataset_path, img_size = (256, 256), num_images = 'all'):

    ImageFile.LOAD_TRUNCATED_IMAGES = True
    
    MAPILLARY_DATASET = dataset_path
    
    #Parent Directory
    parent_folder = 'resized_' + str(img_size[0])
    parent_folder_path = os.path.join(MAPILLARY_DATASET,parent_folder)
    try:
        os.makedirs(parent_folder_path)
    except:
        print('Folder', parent_folder, 'already exists')
        
    
    # Subdirectories
    folders = ['training/images', 'training/labels', 'validation/images', 'validation/labels', 'testing/images']
    for folder in folders:
        try:
            os.makedirs(os.path.join(MAPILLARY_DATASET,parent_folder, folder))
        except:
            print('Folder', folder, 'already exists')
            print('Please delete the directory and try again.')
            return
        
    
    print("New Parent Directory Located at:", parent_folder_path)
    
    
    for folder in folders:
    folder_path = os.path.join(MAPILLARY_DATASET, folder)
    filenames = sorted(os.listdir(folder_path))
    
    
    if num_images == 'all':
        
        for file in filenames:

            file_path = os.path.join(folder_path, file)
            image = Image.open(file_path).resize(img_size)
            image.save(os.path.join(parent_folder_path, folder, file))
    else:
        
        for file in filenames:

            file_path = os.path.join(folder_path, file)
            image = Image.open(file_path).resize(img_size)
            image.save(os.path.join(parent_folder_path, folder, file))

            
    return parent_folder_path
    