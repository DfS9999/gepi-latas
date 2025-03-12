from argparse    import ArgumentParser
from os          import mkdir, listdir
from time        import strftime
from random      import seed, randint, choice
from PIL         import Image
from torchvision import transforms
from PIL         import ImageDraw

CARD_SIZE     = 224
TARGET_SIZE   = 640

CLASS_IDX_TABLE = {
    'makk_7': 0, 'makk_8': 1, 'makk_9': 2, 'makk_10': 3,
    'makk_also': 4, 'makk_felso': 5, 'makk_kiraly': 6, 'makk_asz': 7,
    'sziv_7': 8, 'sziv_8': 9, 'sziv_9': 10, 'sziv_10': 11,
    'sziv_also': 12, 'sziv_felso': 13, 'sziv_kiraly': 14, 'sziv_asz': 15,
    'tok_7': 16, 'tok_8': 17, 'tok_9': 18, 'tok_10': 19,
    'tok_also': 20, 'tok_felso': 21, 'tok_kiraly': 22, 'tok_asz': 23,
    'zold_7': 24, 'zold_8': 25, 'zold_9': 26, 'zold_10': 27,
    'zold_also': 28, 'zold_felso': 29, 'zold_kiraly': 30, 'zold_asz': 31
}

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
    output_labels_path = fr'{output_root_path}\labels'
    mkdir(path=output_imgs_path)
    mkdir(path=output_labels_path)
    print("Created new folder for the dataset: ", output_root_path)
    
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
    
    new_card_size   = CARD_SIZE - randint((int)(-CARD_SIZE*0.1), (int)(CARD_SIZE*0.1))
    transform_cards = transforms.Compose([
        transforms.RandomRotation(degrees=45),
        transforms.Resize(size=((new_card_size, new_card_size)))
    ])
    
    # x,y is upper left
    placement_min = 0 - (int)(CARD_SIZE * 0.4)
    placement_max = TARGET_SIZE - (int)(CARD_SIZE * 0.4)
    
    output_extension = "jpg"
    
    for i in range(arguments.count):
        output_id         = f"{i:>08}"
        output_img_path   = fr"{output_imgs_path}\{output_id}.{output_extension}"
        output_label_path = fr"{output_labels_path}\{output_id}.txt"
        
        bg = choice(bg_imgs).copy()
        
        output_img = transform_bgs(bg)
        with open(output_label_path, 'a') as label_file: 
            for j in range(randint(2, 4)):
                class_name = choice(class_names)
                class_path = fr"{images_root_folder}\{class_name}"   

                card_img_names = listdir(class_path)
                card_img_name  = choice(card_img_names)
                card_img_path  = fr"{class_path}\{card_img_name}"

                card_img = Image.open(card_img_path).copy()            

                transformed_card_img = transform_cards(card_img)

                x, y = [randint(placement_min, placement_max) for _ in range(2)]
                output_img.paste(
                    im=transformed_card_img, 
                    box=(x, y),
                    mask=transformed_card_img.getchannel('A'))

                """
                One row per object. Each row is | class | x_center | y_center | width | height |
                format. Box coordinates must be in normalized xywh format (from 0 to 1). 
                If your boxes are in pixels, divide x_center and width by image width, and y_center and height by image height.
                """
                cls      = CLASS_IDX_TABLE[class_name]
                x_center = (x + transformed_card_img.width / 2)  / TARGET_SIZE
                y_center = (y + transformed_card_img.height / 2) / TARGET_SIZE
                width    = transformed_card_img.width / TARGET_SIZE
                height   = transformed_card_img.height / TARGET_SIZE
                label_text = f"{cls} {x_center} {y_center} {width} {height}\n"
                label_file.write(label_text)

                # bounding box drawing
# TODO get the real values
                ImageDraw.Draw(output_img).rectangle(
                    xy=[x, y, 
                        x + transformed_card_img.width, 
                        y + transformed_card_img.height],
                    outline=(255,0,0),
                    width=3
                )

            output_img.convert('RGB').save(output_img_path)
            print(f"image saved: {output_img_path}")

if __name__ == "__main__":
    main()
