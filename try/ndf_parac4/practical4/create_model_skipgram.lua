require 'nn'
require 'requ'

create_model = {}

function create_model.create_model_skipgram(opt)
  ------------------------------------------------------------------------------
  -- MODEL
  ------------------------------------------------------------------------------
  -- OUR MODEL:
  local model = nn.Sequential()
  -- local inputs = nn.ParallelTable()
  local context_vecs = nn.LookupTable(opt.vocab_size, opt.dim)
  local current_vec = nn.LookupTable(opt.vocab_size, opt.dim)
  -- inputs:add(context_vecs)
  -- inputs:add(current_vec)
  -- model:add(inputs)

  model:add(nn.ParallelTable())
  model.modules[1]:add(context_vecs)
  model.modules[1]:add(current_vec)
  model:add(nn.MM(false, true))
  model:add(nn.Sigmoid())

  ------------------------------------------------------------------------------
  -- LOSS FUNCTION
  ------------------------------------------------------------------------------
  local criterion = nn.BCECriterion()

  return model, criterion
end

return create_model

