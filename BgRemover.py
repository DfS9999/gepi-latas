from argparse import ArgumentParser
from time     import strftime
from os       import mkdir, listdir
from rembg    import new_session, remove

def main():
    parser = ArgumentParser()
    parser.add_argument('root_folder_structure', type=str)
    parser.add_argument('--model', default="birefnet-general", type=str)
    arguments = parser.parse_args()
    root = arguments.root_folder_structure[:-1]   # remove ending '\'
    
    bgrm_root = root + "_bgrm" + strftime('d%H%M%S')
    mkdir(path=bgrm_root)
    print("Created new root folder: ", bgrm_root)
    
#    session = new_session(model_name="u2net")
    session = new_session(model_name=arguments.model)
    
    for class_folder in listdir(root):
        mkdir(path=fr"{bgrm_root}\{class_folder}")
        print("Created new class folder:", class_folder)

        for picture_name in listdir(fr"{root}\{class_folder}"):
            original_picture_path = fr"{root}\{class_folder}\{picture_name}"
            bgrm_picture_path = fr"{bgrm_root}\{class_folder}\{picture_name}"
            print(f"Removing background from [{picture_name}] ... ", end='')
            with open(original_picture_path, 'rb') as og:
                with open(bgrm_picture_path, 'wb') as bgrm:
                    og_content = og.read()
                    result = remove(data=og_content, session=session)
                    bgrm.write(result)
            print("Done!")
 
if "__main__" == __name__:
    main()
