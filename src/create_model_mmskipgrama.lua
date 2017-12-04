require 'nn'
require 'MMCriterion'

local mmskipgrama_model = {
    model = nn.Sequential(),
    criterion = nn.MMCriterion(),
    vec = nn.LookupTable(self.vocab_size, self.dim)
}

function mmskipgrama_model:create_model_mmskipgrama(opt)
  ------------------------------------------------------------------------------
  -- MODEL
  ------------------------------------------------------------------------------
  self.model = nn.Sequential()
  local inputs = nn.ParallelTable()
  self.vocab_size = 100
  self.dim = 4
  inputs:add(nn.Identity())
  inputs:add(nn.Identity())
  inputs:add(nn.Identity())
  self.model:add(inputs)

  ------------------------------------------------------------------------------
  -- LOSS FUNCTION
  ------------------------------------------------------------------------------
  self.criterion = nn.TripletCriterion()

  print(self)
  return self.model, self.criterion
end

return mmskipgrama_model

