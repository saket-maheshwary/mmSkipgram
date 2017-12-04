require 'nn'
require 'MMCriterion'

local MultiModalSkipgram = {}

function MultiModalSkipgram.create_model_skipgram(opt)

  local current_vec = nn.LookupTable(opt.vocab_size, opt.dim)
  local context_vecs = nn.LookupTable(opt.vocab_size, opt.dim)
  local current_vec_img = current_vec:clone('weight','gradWeight')
  -- WORD2VEC MODEL
  ------------------------------------------------------------------------------

  local w2v = nn.Sequential()
  w2v:add(nn.ParallelTable())
  w2v.modules[1]:add(context_vecs)
  w2v.modules[1]:add(current_vec)
  w2v:add(nn.MM(false, true))
  w2v:add(nn.Sigmoid())

  ------------------------------------------------------------------------------
  -- WORD2VEC CRITERION
  ------------------------------------------------------------------------------
  local w2v_criterion = nn.BCECriterion()

  ------------------------------------------------------------------------------
  -- WORD2IMG MODEL
  ------------------------------------------------------------------------------

  local w2i = nn.Sequential()
  local w2i_inputs = nn.ParallelTable()
  w2i_inputs:add(current_vec_img)
  w2i_inputs:add(nn.Identity())
  w2i_inputs:add(nn.Identity())
 -- w2i_inputs:add(nn.Linear(opt.imgdim,opt.dim))
 -- w2i_inputs:add(nn.Linear(opt.imgdim,opt.dim))
  w2i:add(w2i_inputs)

  ------------------------------------------------------------------------------
  -- WORD2IMG CRITERION
  ------------------------------------------------------------------------------

--  local w2i_criterion = nn.TripletCriterion() 
  local w2i_criterion = nn.MMCriterion()
  ------------------------------------------------------------------------------
  -- MODEL
  ------------------------------------------------------------------------------

  local model = nn.Sequential()
  model:add(nn.ParallelTable())
  model.modules[1]:add(w2v)
  model.modules[1]:add(w2i)

  ------------------------------------------------------------------------------
  -- CRITERION
  ------------------------------------------------------------------------------

  local criterion = nn.ParallelCriterion()
  criterion:add(w2v_criterion)
  criterion:add(w2i_criterion)

  if opt.cuda == 1 then
        require("cunn")
	require("cutorch")
	cutorch.setDevice(1)
	criterion:cuda()
--	w2v_criterion:cuda()
--	w2i_criterion:cuda()
	model:cuda()
--	w2v:cuda()
--	w2i:cuda()
  end	
  return model, criterion, current_vec
end

return MultiModalSkipgram

