# split into train and test set
from os import listdir
from xml.etree import ElementTree
from numpy import zeros
from numpy import asarray
from mrcnn.utils import Dataset

# class that defines and loads the kangaroo dataset
class HelmetDataset(Dataset):
	# load the dataset definitions
	im_id=1                                    # image id starting from 1
	def load_dataset(self, dataset_dir, is_train=True):
		# define one class
		self.add_class("dataset", 1, "helmet")
		self.add_class("dataset",2,"person with helmet")
		self.add_class("dataset",3,"person without helmet")
		# define data locations,
		if is_train:
			images_dir = dataset_dir + '/train_image_folder/'
		else:
			images_dir = dataset_dir + '/test_image_folder/'
		annotations_dir = dataset_dir + '/annot_folder/'
		# find all images
		for filename in listdir(images_dir):
			# image id, we get from the class variable "im_id"
			image_id = HelmetDataset.im_id
			img_path = images_dir + filename
			ann_path = annotations_dir + str(image_id) + '.xml'
			# add to dataset
			self.add_image('dataset', image_id=image_id, path=img_path, annotation=ann_path)
			HelmetDataset.im_id+=1

	# extract bounding boxes from an annotation file
	def extract_boxes(self, filename):
		# load and parse the file
		tree = ElementTree.parse(filename)
		# get the root of the document
		root = tree.getroot()
		# extract each bounding box
		boxes = list()
		for box in root.findall('.//bndbox'):
			xmin = int(box.find('xmin').text)
			ymin = int(box.find('ymin').text)
			xmax = int(box.find('xmax').text)
			ymax = int(box.find('ymax').text)
			coors = [xmin, ymin, xmax, ymax]
			boxes.append(coors)
		# extract image dimensions
		width = int(root.find('.//size/width').text)
		height = int(root.find('.//size/height').text)
		return boxes, width, height

	# load the masks for an image
	def load_mask(self, image_id):
		# get details of image
		info = self.image_info[image_id]
		# define box file location
		path = info['annotation']
		# load XML
		boxes, w, h = self.extract_boxes(path)
		# create one array for all masks, each on a different channel
		masks = zeros([h, w, len(boxes)], dtype='uint8')
		# create masks
		class_ids = list()
		for i in range(len(boxes)):
			box = boxes[i]
			row_s, row_e = box[1], box[3]
			col_s, col_e = box[0], box[2]
			masks[row_s:row_e, col_s:col_e, i] = 1
			class_ids.append(self.class_names.index('kangaroo'))
		return masks, asarray(class_ids, dtype='int32')

	# load an image reference
	def image_reference(self, image_id):
		info = self.image_info[image_id]
		return info['path']

# train set
train_set = HelmetDataset()
train_set.load_dataset('github_hard_hat', is_train=True)
train_set.prepare()
print('Train: %d' % len(train_set.image_ids))

# test/val set
test_set = HelmetDataset()
test_set.load_dataset('github_hard_hat', is_train=False)
test_set.prepare()
print('Test: %d' % len(test_set.image_ids))
