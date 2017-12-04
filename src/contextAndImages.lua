local context = {}
local opt = require 'opt'
if opt.cuda == 1 then
	  require("cunn")
	  require("cutorch")	
end
function context.get_context(data, mb_start, mb_end)

  local inputs = {}
  local contextids = {}
  local wordid = {}
  local contextid_minibatch = {}
  local wordid_minibatch = {}
  local minibatch_sz = mb_end - mb_start + 1 
  local wordimgFV_minibatch = (opt.cuda == 1) and torch.CudaTensor(minibatch_sz, opt.imgdim) or  torch.Tensor(minibatch_sz, opt.imgdim)
  local wordnegimgFV_minibatch = (opt.cuda == 1) and torch.CudaTensor(minibatch_sz, data.negImgSize,opt.imgdim) or torch.Tensor(minibatch_sz, data.negImgSize,opt.imgdim)
  local targets = {}
  local labels = {}
  local index = 1
  local imageWordsSeen = 0
  local m = 1

  for i = mb_start, mb_end do
        if i > data.contextSize / 2 then
            contextids = {}
            for j = i - data.contextSize / 2, i + data.contextSize / 2 do
                if j ~= i then
                    contextids[#contextids + 1] = data.word2id[data.text[j]]
                    targets[#targets + 1] = 1
                end
            end
            -- print(targets)
            wordid = data.word2id[data.text[i]]
            k = 0
            while k < data.contextSize do
                found = false
                c = torch.random(data.numDistictWords)
                for l = 1, #contextids do
                    if contextids[l] == c then
                        found = true
                        break
                    end
                end
                if found == false then
                      contextids[#contextids + 1] = c
                      targets[#targets + 1] = 0
                      k = k+1
                end
           end
	   found = false
	   m = 1
	   for m = 1, #data.imgWords do
	      if data.imgWords[m] == data.text[i] then
  	     	found = true
		break
	      end 
	   end
	   wordImgFV = (opt.cuda == 1) and torch.CudaTensor(1,opt.imgdim):zero() or torch.zeros(1,opt.imgdim)
	   wordNegImgFV = (opt.cuda == 1) and torch.CudaTensor(data.negImgSize,opt.imgdim):zero() or torch.zeros(data.negImgSize,opt.imgdim)
	   if found == true then
		imageWordsSeen = imageWordsSeen + 1
		wordImgFV = data.imgFV[{{m},{1,opt.imgdim}}]
		k = 1
		while k <= data.negImgSize do
		 c = torch.random(#data.imgWords)
		 if c ~= l then
		  wordNegImgFV[{{k},{}}] = data.imgFV[{{c},{1,opt.imgdim}}]
		  k = k + 1
		 end
		end
	   end
           -- contextids = torch.Tensor(contextids)
	   -- print(i)
            contextid_minibatch[#contextid_minibatch + 1] = contextids
            wordid_minibatch[#wordid_minibatch + 1] = wordid
	    wordimgFV_minibatch[{{index},{}}] = wordImgFV
	    wordnegimgFV_minibatch[{{index},{},{}}] = wordNegImgFV
            -- inputs[#inputs + 1] = {contextids, word2id}
            labels[#labels + 1] = targets
            wordid = {}
            contextids = {}
            targets = {}
	    index = index + 1
            -- collectgarbage()
        end
  end
  local cid = torch.Tensor(mb_end-mb_start+1, 2*data.contextSize)
  --local cid = torch.Tensor(2*data.contextSize, mb_end-mb_start+1)
  local wid = torch.Tensor(mb_end-mb_start+1, 1)
  --local lab = torch.IntTensor(2*data.contextSize,mb_end-mb_start+1)
  local lab = torch.Tensor(mb_end-mb_start+1, 2*data.contextSize)
  local m = 1
  local n = 1 
  for mk, mv in pairs(contextid_minibatch) do
      n = 1
      for nk, nv in pairs(contextid_minibatch[mk]) do
           cid[{{m}, {n}}] = contextid_minibatch[mk][nk]
           lab[{{m},{n}}] = labels[mk][nk]
           n = n + 1
      end
      wid[{{m},{1}}] = wordid_minibatch[mk]
      m = m + 1
  end
  --inputs = {cid, wid}
  --print(cid)
  --print(wid)
  --print(contextid_minibatch) 
  -- inputs = {contextid_minibatch,wordid_minibatch}
  --inputs = torch.Tensor(inputs)
  --labels = torch.Tensor(labels)
  --print(inputs)
  -- print(lab)
  -- return inputs, labels
  --print(wordimgFV_minibatch:type())
  --print(wordnegimgFV_minibatch:type())

  -- print('Image words seen:', imageWordsSeen, ' minibatch size:', minibatch_sz)
  return cid, wid, lab, wordimgFV_minibatch, wordnegimgFV_minibatch
  --return {contextid_minibatch, wordid_minibatch}
end

return context
