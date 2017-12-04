require 'nn'
require 'requ'

function create_model_skipgram(opt)
  ------------------------------------------------------------------------------
  -- MODEL
  ------------------------------------------------------------------------------
  -- OUR MODEL:
  local model = nn.Sequential()
  local inputs = nn.ParallelTable()
  inputs:add(opt.context_vecs)
  inputs:add(opt.current_vec)
  model:add(inputs)
  model:add(nn.MM(false, true))
  model:add(nn.Sigmoid())

  ------------------------------------------------------------------------------
  -- LOSS FUNCTION
  ------------------------------------------------------------------------------
  local criterion = nn.BCECriterion()

  return model, criterion
end

return create_model

