from argparse    import ArgumentParser
from os          import mkdir, listdir
from time        import strftime
from random      import seed, randint, choice
from PIL         import Image
from torchvision import transforms

CARD_SIZE     = 224
TARGET_SIZE   = 640

def main():
    parser = ArgumentParser()
    parser.add_argument('count', type=int)
    parser.add_argument('images_root_folder', type=str)
    parser.add_argument('bg_images_folder', type=str)
    parser.add_argument('--seed', default=None, type=int)
    arguments = parser.parse_args()
    images_root_folder = arguments.images_root_folder[:-1]
    bg_imgs_folder     = arguments.bg_images_folder[:-1]

    seed(None)
    output_root_path = fr'.\dataset{randint(100,999)}_{strftime('%d%H%M%S')}'
    seed(arguments.seed)
    mkdir(path=output_root_path)
    output_imgs_path = fr'{output_root_path}\images'
    mkdir(path=output_imgs_path)
    print("Created new folder for the dataset: ", output_root_path)
    
# TODO create yaml file 

    bg_names = listdir(bg_imgs_folder)
    bg_imgs  = [Image.open(fr"{bg_imgs_folder}\{bg_name}") for bg_name in bg_names]
    print(f"Loaded {len(bg_imgs)} background images.")

    class_names = listdir(images_root_folder)
    
    transform_bgs = transforms.Compose([
        transforms.RandomChoice([
            lambda img: transforms.functional.rotate(img=img, angle=0),
            lambda img: transforms.functional.rotate(img=img, angle=90),
            lambda img: transforms.functional.rotate(img=img, angle=180),
            lambda img: transforms.functional.rotate(img=img, angle=270),
        ])
    ])
    
    new_card_size   = CARD_SIZE - randint(-16, 16)
    transform_cards = transforms.Compose([
        transforms.RandomRotation(degrees=45),
        transforms.Resize(size=((new_card_size, new_card_size)))
    ])
    
    # x,y is upper left
    placement_min = 0 - (CARD_SIZE // 2)
    placement_max = TARGET_SIZE - (CARD_SIZE // 2)
    
    output_extension = "jpg"
    
    for i in range(arguments.count):
        output_img_name = f"img_{i:>05}.{output_extension}"
        output_img_path = fr"{output_imgs_path}\{output_img_name}"
        
        bg = choice(bg_imgs).copy()
        
        output_img = transform_bgs(bg)

        for j in range(randint(2, 4)):
            class_name = choice(class_names)
            class_path = fr"{images_root_folder}\{class_name}"   

            card_img_names = listdir(class_path)
            card_img_name  = choice(card_img_names)
            card_img_path  = fr"{class_path}\{card_img_name}"
            
            card_img = Image.open(card_img_path).copy()            
            
            transformed_card_img = transform_cards(card_img)
            
            x, y = [randint(placement_min, placement_max) for _ in range(2)]

# TODO calculate bounding box
# TODO create yolo-yaml file, and write :
#       - output_img_name 
#       - each card's class, positon, bounding box


            output_img.paste(
                im=transformed_card_img, 
                box=(x, y),
                mask=transformed_card_img.getchannel('A'))
            print(f"{i}/{j}. image pasted onto background at x:{x} y:{y}")            

        output_img.convert('RGB').save(output_img_path)
        print(f"image saved: {output_img_path}")

if __name__ == "__main__":
    main()
