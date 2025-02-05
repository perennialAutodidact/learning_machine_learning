import re
from pathlib import Path
import os
import shutil


def format_filename(filename, index, level=0):
    snake_pattern = re.compile('\w+_{1}\w+', re.IGNORECASE)
    filename = filename.lower()

    already_snake = snake_pattern.match(filename)

    if already_snake:
        return filename.lower()
    
    formatted_name = ''

    segments = [seg for seg in filename.lower().split(' ') if not re.search('([()]|[-]{2,})', seg)]
    if '-' in segments:
        chop_index = segments.index('-') + 1
        segments = segments[chop_index:]

    formatted_name += '_'.join(segments).replace('-', '_')

    if not re.match('\d', formatted_name) and formatted_name not in ['python', 'r'] and not formatted_name.startswith('color'):
        formatted_name = f"{(index + 1):02d}_{formatted_name}"

    return formatted_name if formatted_name  else filename.lower()
    

def clean_directory(root, level=0):
    for parent_dir, dirs, files in os.walk(root):
        for index, dirname in enumerate(sorted(dirs)):
            dir = os.path.join(parent_dir, dirname)
            if os.path.isdir(dir) and not os.listdir(dir):
                os.rmdir(dir)
            else:
                if dirname.lower() == 'r':
                    for file in os.listdir(dir):
                        if not os.path.isdir(os.path.join(dir, file)):
                            os.remove(os.path.join(dir, file))
                        else:
                            for file in os.listdir(dir):
                                filename = os.path.join(dir, file)
                                for file in os.listdir(filename):
                                    os.remove(os.path.join(filename, file))

                if dirname.lower() == 'python':
                    if os.path.isdir(parent_dir) and not parent_dir == str(root):
                        for file in os.listdir(dir):
                            filename = os.path.join(dir, file)
                            file = Path(filename)
                            if file.name == 'color_blind_friendly_images':
                                shutil.move(file, file.parent.parent)
                            if file.suffix in ['.ipynb']:
                                os.remove(filename)

                            # print(f"moving {filename} to {parent_dir}")
                            # shutil.move(filename, parent_dir)
                
                original_filename = dir
                formatted_filename = os.path.join(parent_dir, format_filename(dirname, index, level))

                # print(f"{original_filename} -> {formatted_filename}")
                os.rename(original_filename, formatted_filename)

                clean_directory(os.path.join(parent_dir, dir), level + 1)

        for index, filename in enumerate(files):
            parent = Path(parent_dir)
            file = Path(os.path.join(parent_dir, filename))

            if file.suffix in ['.R', '.ipynb'] or file.name in ['.Rhistory', '.DS_Store']:
                os.remove(file)
            else:
                new_filename = os.path.join(parent.parent, filename.lower())
                if parent.name.lower() == 'python':
                    shutil.move(file, new_filename)
       

            

    #         subdirs = [file.name for file in item.glob('**/**') if file.is_dir()]
    #         print(item.name, subdirs)
    #         if subdirs:
    #             clean_directory(item, level + 1)
    #         else:
    #             # if not os.listdir(item.name):
    #             # print('EMPTY:', format_filename(item.name))

    #             print('dir contents:', os.listdir(item.name))
                   
        # else:
        #     # delete a file: item.unlink()
        #     if item.name in ['.Rhistory', '.DS_Store'] or item.suffix in ['.R','.ipynb']:
        #         item.unlink()
        #     elif item.suffix in ['.csv', '.tsv'] and item.parents[0].name == 'R':
        #         item.unlink()
        #     else:
        #         formatted_name = format_filename(item.name)    
        #         new_name = item / Path(formatted_name)
        #         # item.rename(new_name)
        #         print(f"RENAME: {item.name} to {new_name}")

 

def main():
    root = Path(Path.cwd()) / 'src'
    clean_directory(root)
    
    

if __name__ == '__main__':
    main()