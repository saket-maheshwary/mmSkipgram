npy4th = require 'npy4th'

local ImageVectors = {}

--------------------------------------------------------------------------------
local wordsWithImagesList = '/data5/sourabh.d/SelectedImages_output/ImageNet/cnn0/flist_dir/words.txt'
local featuresDir = '/data5/sourabh.d/SelectedImages_output/ImageNet/cnn0/alexnet_features'
local numImagesPerWord = 50
local fvDim = 4096

--------------------------------------------------------------------------------

function trim (s)
      return (string.gsub(s, "^%s*(.-)%s*$", "%1"))
end

--------------------------------------------------------------------------------

function ImageVectors.getImageVectors()
	-- read input file to create word list
	local words = {}
	wfd, werr = io.open(wordsWithImagesList)
	if werr then print('Cannot open ', wordsWithImagesList, '\n'); return ; end
	while true do
		line = wfd:read()
		if line == nil then break end
		words[#words + 1] = trim(line)
	end 
--	print(words)

	imgFV = torch.Tensor(#words, fvDim)
	for i = 1, #words do
	   npyfile = string.format('%s/%s.npy', featuresDir, words[i])
	  -- print(npyfile)
	   fv = npy4th.loadnpy(npyfile)
	   imgFV[{{i},{}}] = fv:mean(1)
	end
	return imgFV, words
end


function ImageVectors.getAllImageVectors()
	-- read input file to create word list
	local bsize = 1
	local numNeg = 4
	local words = {}
	wfd, werr = io.open(wordsWithImagesList)
	if werr then print('Cannot open ', wordsWithImagesList, '\n'); return ; end
	while true do
		line = wfd:read()
		if line == nil then break end
		words[#words + 1] = trim(line)
	end 
--	print(words)

	imgFVAll = torch.Tensor(#words, numImagesPerWord, fvDim)
	for i = 1, #words do
	   npyfile = string.format('%s/%s.npy', featuresDir, words[i])
	--   print(npyfile)
	   fv = npy4th.loadnpy(npyfile)
	   imgFVAll[{{i},{},{}}] = fv
	end
	
	local R = torch.randperm(#words)
	local S = torch.randperm(numImagesPerWord)
	local i = 1
	local j = 1
	local i1 = 0
	local j1 = 0
	local j2 = 0
	local j3 = 0 

	local a = torch.Tensor(bsize, fvDim)
	local p = torch.Tensor(bsize, fvDim)
	local n = torch.Tensor(bsize, numNeg, fvDim)
	
	for k = 1, bsize do	
		i1 = R[i]
		local i2 = {}
		for l = 1, numNeg do
		      i2[#i2+1] = R[i+l]
		end
		j1 = S[j]
		j2 = S[j+1]
		j3 = S[j+2]
		a[{{k}, {}}] = imgFVAll[{{i1}, {j1}, {}}]
		p[{{k}, {}}] = imgFVAll[{{i1}, {j2}, {}}]
		n[{{k}, {}, {}}] = torch.squeeze(imgFVAll[{{}, {j3}, {}}]:index(1,torch.LongTensor(i2)),2)
		i = i + 5
		j = j + 3
	end

	return {a, p, n}
end


return ImageVectors
