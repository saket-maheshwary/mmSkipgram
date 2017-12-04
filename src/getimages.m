%clear
%clc


% get N number of images for each of selected words
path = '/Pulsar1/Datasets/Imagenet/';
mkdir SelectedImages
load('words.mat');
images = {};
N = 50;

for i=1:size(ids,1)
	words(i,:)
	name = ids(i,:);
	name = [name  '.tar'];
	mkdir output
	untar([path name],'./output/');
	filenames = dir('./output/');
	filenames = filenames(3:end);
	x = randperm(size(filenames,1));
	x = x(1:N);
	filenames = filenames(x);
	img_nms = {};
	for j=1:N
		img_nms{j}=filenames(j).name
		copyfile(['./output/' img_nms{j}],'./SelectedImages/');
	end
	rmdir('output','s');
	images{i} = img_nms;
end

save('images.mat','images');


