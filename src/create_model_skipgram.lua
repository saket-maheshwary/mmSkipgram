require 'nn'

create_model = {}

function create_model.create_model_skipgram(opt)
  ------------------------------------------------------------------------------
  -- MODEL
  ------------------------------------------------------------------------------
  -- OUR MODEL:
  local model = nn.Sequential()
  -- local inputs = nn.ParallelTable()
  -- inputs:add(context_vecs)
  -- inputs:add(current_vec)
  -- model:add(inputs)

  local current_vec = nn.LookupTable(opt.vocab_size, opt.dim)
  local context_vecs = nn.LookupTable(opt.vocab_size, opt.dim)
--  current_vec:reset(0.25)
--  context_vecs:reset(0.25)
  model:add(nn.ParallelTable())
  model.modules[1]:add(context_vecs)
  model.modules[1]:add(current_vec)
  model:add(nn.MM(false, true))
  model:add(nn.Sigmoid())

  ------------------------------------------------------------------------------
  -- LOSS FUNCTION
  ------------------------------------------------------------------------------
  local criterion = nn.BCECriterion()
  if opt.cuda == 1 then
	require("cunn")
	require("cutorch")
	cutorch.setDevice(1)
	criterion:cuda()
	model:cuda()
  end
  return model, criterion, current_vec
end

return create_model

