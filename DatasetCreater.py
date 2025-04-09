from argparse    import ArgumentParser
from os          import mkdir, listdir
from time        import strftime
from PIL         import Image, ImageDraw
from torchvision import transforms
import random
import cv2
import numpy as np

DRAW_BOUNDING_BOXES = False
CARD_SIZE        = 224
DATASET_IMG_SIZE = 640
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

def ReadArguments():
    parser = ArgumentParser()
    parser.add_argument('images_root_folder', type=str)
    parser.add_argument('bg_images_folder', type=str)
    parser.add_argument('dataset_size', type=int)
    parser.add_argument('max_overlap_percentage', type=float)
    parser.add_argument('--seed', default=None, type=int)
    return parser.parse_args()

def CreateOutputDirs():
    output_root_path   = fr".\dataset_{strftime('%y%m%d%H%M%S')}"
    output_imgs_path   = fr'{output_root_path}\images'
    output_labels_path = fr'{output_root_path}\labels'
    mkdir(path=output_root_path)
    mkdir(path=output_imgs_path)
    mkdir(path=output_labels_path)
    print("Created new folder for the dataset: ", output_root_path)
    return output_imgs_path, output_labels_path

def LoadImagesFromDir(directory):
    img_names = listdir(directory)
    return [Image.open(fr"{directory}\{img_name}") for img_name in img_names]

def DrawStandingBoundingBox(img, box_x, box_y, box_width, box_height):
    ImageDraw.Draw(img).rectangle(
        xy=[box_x, box_y, box_x + box_width, box_y + box_height],
        outline=(255,0,0), width=3)

def main():
    # arguments
    arguments = ReadArguments()
    imgs_root_folder    = arguments.images_root_folder.rstrip(r"\/")
    bg_imgs_folder      = arguments.bg_images_folder.rstrip(r"\/")
    dataset_size        = arguments.dataset_size
    overlap_percentage  = arguments.max_overlap_percentage

    random.seed(arguments.seed)

    class_names   = listdir(imgs_root_folder)

    # create output directory
    output_imgs_path, output_labels_path = CreateOutputDirs()
    
    # load background images
    bg_imgs = LoadImagesFromDir(bg_imgs_folder)
    
    # transforms
    bg_transforms = transforms.Compose([
        transforms.RandomChoice([
            lambda bg: transforms.functional.rotate(img=bg, angle=0),
            lambda bg: transforms.functional.rotate(img=bg, angle=90),
            lambda bg: transforms.functional.rotate(img=bg, angle=180),
            lambda bg: transforms.functional.rotate(img=bg, angle=270),
        ]),
        transforms.Resize(
            size=DATASET_IMG_SIZE, 
            interpolation=transforms.InterpolationMode.BICUBIC)
    ])
    
    card_transforms = transforms.Compose([
        transforms.RandomRotation(degrees=60),
    ])
    
    for i in range(dataset_size):
        output_id         = f"{i:>05}"
        output_img_path   = fr"{output_imgs_path}\{output_id}.jpg"
        output_label_path = fr"{output_labels_path}\{output_id}.txt"

        bg = random.choice(bg_imgs).copy()
        output_img = bg_transforms(bg)
        
        with open(output_label_path, 'w') as label_file: 
            previous = []
            for _ in range(random.randint(2, 5)):
                class_name     = random.choice(class_names)
                class_path     = fr"{imgs_root_folder}\{class_name}"
                card_img_names = listdir(class_path)
                card_img_name  = random.choice(card_img_names)
                card_img_path  = fr"{class_path}\{card_img_name}"
                
                with Image.open(card_img_path) as card_img:
                    transformed_card_img = card_transforms(card_img.copy())
                alpha_card   = transformed_card_img.getchannel('A')
                alpha_pixels = cv2.findNonZero(np.array(alpha_card))
                standing_box_x, standing_box_y, standing_box_w, standing_box_h = cv2.boundingRect(alpha_pixels) # (x,y) top-left
                (rot_box_x, rot_box_y), (rot_box_w, rot_box_h), rot_box_r = cv2.minAreaRect(alpha_pixels) # returns: (x,y),(w,h),r
                
                for _ in range(40):
                    img_x, img_y    = [random.randint(0, DATASET_IMG_SIZE - CARD_SIZE) for _ in range(2)]
                    shifted_standing_x = standing_box_x + img_x
                    shifted_standing_y = standing_box_y + img_y
                    shifted_rot_x   = rot_box_x + img_x
                    shifted_rot_y   = rot_box_y + img_y
                    shifted_rot_box = (shifted_rot_x, shifted_rot_y), (rot_box_w, rot_box_h), rot_box_r
                    
                    area = rot_box_w * rot_box_h
                    if (area <= 0):
                        continue
                    
                    # check overlapping
                    successful_placement = True
                    for p in previous:
                        p_box = p[0]
                        p_area = p[1]
                        
                        overlap_area = 0.0
                        intersection, region = cv2.rotatedRectangleIntersection(p_box, shifted_rot_box)
                        if (intersection != cv2.INTERSECT_NONE) and (region is not None):
                            overlap_area = cv2.contourArea(region)
                        
                        if (overlap_area / p_area > overlap_percentage):
                            successful_placement = False
                            break
                    
                    if (successful_placement):
                        previous.append((shifted_rot_box, area))
                        output_img.paste(im=transformed_card_img, box=(img_x, img_y), mask=alpha_card)
                        
                        # label data
                        class_idx         = CLASS_IDX_TABLE[class_name]
                        norm_box_center_x = (shifted_standing_x + standing_box_w / 2) / DATASET_IMG_SIZE
                        norm_box_center_y = (shifted_standing_y + standing_box_h / 2) / DATASET_IMG_SIZE
                        norm_width        = standing_box_w / DATASET_IMG_SIZE
                        norm_height       = standing_box_h / DATASET_IMG_SIZE
                        label_text = f"{class_idx} {norm_box_center_x} {norm_box_center_y} {norm_width} {norm_height}\n"
                        label_file.write(label_text)

                        if (DRAW_BOUNDING_BOXES):
                            DrawStandingBoundingBox(output_img, shifted_standing_x, shifted_standing_y, standing_box_w, standing_box_h)
                            output_img_np = np.array(output_img)
                            cv2.drawContours(output_img_np, [np.int32(cv2.boxPoints(shifted_rot_box))], 0, (0,0,255), 3)
                            output_img = Image.fromarray(output_img_np)
                        break
                
            output_img.convert('RGB').save(output_img_path)
            print(f"image saved: {output_img_path}")

if __name__ == "__main__":
    main()
