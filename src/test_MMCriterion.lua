require 'MMCriterion'
local ImageVectors = require 'ImageVectors'

local function jacobian_wrt_input(module, x, eps)
  
  local z = module:updateOutput(x)
  local grad = module:updateGradInput(x)

  grad_est = {}
  for k = 1, #x do              -- number of inputs
      grad_est[#grad_est + 1] = torch.zeros(x[k]:size()):clone()
      for l = 1, x[k]:size()[1] do
	for i = 1, x[k]:size()[2] do -- dimensions
            x[k][{{l}, {i}}] = x[k][{{l}, {i}}] + eps
            local z_plus = module:updateOutput(x)
          --  module.output = 0
            x[k][{{l}, {i}}] = x[k][{{l}, {i}}] - 2 * eps
            local z_minus = module:updateOutput(x)
          --  module.output = 0
           x[k][{{l}, {i}}] = x[k][{{l}, {i}}] + eps -- important ! restore vector
--            x[k][{{}, {i}}] = x[k][{{}, {i}}] - eps -- important ! restore vector
            -- print(z_plus - z_minus)
            grad_est[k][{{l},{i}}] = (z_plus - z_minus) / (2 * eps)
        end
    end
  end


  for k = 1, #x do
      local d = (grad[k] - grad_est[k]):abs()
      print('Max:', torch.max(d), '\n')
      print('Min:', torch.min(d), '\n')
      print('Mean:', torch.mean(d), '\n')
  end

end
  
torch.manualSeed(1)
inputs = ImageVectors.getAllImageVectors()
print(jacobian_wrt_input(nn.MMCriterion(), inputs, 1e-6))
