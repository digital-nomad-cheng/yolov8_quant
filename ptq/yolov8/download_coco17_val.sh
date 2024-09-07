# Create a directory for the dataset
mkdir -p coco2017

# Download and extract the validation images
wget http://images.cocodataset.org/zips/val2017.zip
unzip val2017.zip -d coco2017

# Download and extract the validation annotations
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip annotations_trainval2017.zip -d coco2017

# Clean up zip files
rm val2017.zip annotations_trainval2017.zip
