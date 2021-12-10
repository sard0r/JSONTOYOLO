import json
import os
import argparse

parser = argparse.ArgumentParser(description='Test yolo data.')
parser.add_argument('-j', help='JSON file', dest='json', default='AY0001013.json', required=False)
parser.add_argument('-o', help='path to output folder', dest='out', default='out2/', required=False)

args = parser.parse_args()

json_file = args.json
output = args.out



class COCO2YOLO:
    def __init__(self):
        self._check_file_and_dir(json_file, output)
        self.labels = json.load(open(json_file, 'r', encoding='utf-8'))
        self.coco_id_name_map = self._categories()
        self.coco_name_list = list(self.coco_id_name_map.values())
        # print("total images", len(self.labels['images']))
        # print("total categories", len(self.labels['categories']))
        # print("total labels", len(self.labels['annotations']))

    def _check_file_and_dir(self, file_path, dir_path):
        if not os.path.exists(file_path):
            raise ValueError("file not found")
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    def _categories(self):
        categories = {}
        for cls in self.labels['categories']:
            categories[cls['id']] = cls['name']
        return categories

    def _load_images_info(self):
        images_info = {}
        for image in self.labels['images']:
            id = image['id']
            file_name = image['file_name']
            if file_name.find('\\') > -1:
                file_name = file_name[file_name.index('\\') + 1:]
            w = 1920
            h = 1080
            images_info[id] = (file_name, w, h)

        return images_info

    def _bbox_2_yolo(self, bbox, img_w, img_h):
        centerx_list = []
        centery_list = []
        w_list = []
        h_list = []
        for i in range(0, len(bbox)):
            x, y, xmax, ymax = bbox[i][0], bbox[i][1], bbox[i][2], bbox[i][3]
            # print(x, y, w, h)
            w = xmax - x
            h = ymax - y
            centerx = (x + xmax) / 2
            centery = (y + ymax) / 2
            dw = 1 / 1920
            dh = 1 / 1080
            centerx *= dw
            w *= dw
            centery *= dh
            h *= dh
            centerx_list.append(centerx)
            centery_list.append(centery)
            w_list.append(w)
            h_list.append(h)
        all_list = [centerx_list, centery_list, w_list, h_list]
        return all_list

    def _convert_anno(self, images_info):
        anno_dict = dict()
        for anno in self.labels['annotations']:
            bbox = anno['bbox']
            image_id = anno['image_id']
            category_id = anno['category_id']

            image_info = images_info.get(image_id)
            image_name = image_info[0]
            img_w = image_info[1]
            img_h = image_info[2]
            yolo_box = self._bbox_2_yolo(bbox, img_w, img_h)
            anno_info = [image_name, category_id, yolo_box]
            anno_infos = anno_dict.get(image_id)
            if not anno_infos:
                anno_dict[image_id] = [anno_info]
            else:
                anno_infos.append(anno_info)
                anno_dict[image_id] = anno_infos
        return anno_dict


    def save_classes(self):
        sorted_classes = list(map(lambda x: x['name'], sorted(self.labels['categories'], key=lambda x: x['id'])))
        # print('coco names', sorted_classes)
        with open('coco.names', 'w', encoding='utf-8') as f:
            for cls in sorted_classes:
                f.write(cls + '\n')
        f.close()

    def coco2yolo(self):
        # print("loading image info...")
        images_info = self._load_images_info()
        # print("loading done, total images", len(images_info))

        # print("start converting...")
        anno_dict = self._convert_anno(images_info)
        # print("converting done, total labels", len(anno_dict))

        # print("saving txt file...")
        self._save_txt(anno_dict)
        # print("saving done")

    def _save_txt(self, anno_dict):
        for k, v in anno_dict.items():
            file_name = v[0][0].split(".")[0] + ".txt"
            f1 = os.path.join(output, file_name)
            for img in v:
                print(len(img))
                #  bbox = [img[2][0],obj[2][1],obj[2][2],obj[2][3]]
                with open(f1, 'w', encoding='utf-8') as f:
                    for obj in range(len(img[1])):
                        f.write(str(img[1][obj]) + ' ' + str(img[2][0][obj]) + ' ' + str(img[2][1][obj]) + ' ' + str(img[2][2][obj]) + ' ' + str(img[2][3][obj]) + '\n')


if __name__ == '__main__':
    c2y = COCO2YOLO()
    c2y.coco2yolo()
