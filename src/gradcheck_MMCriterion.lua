local mmskipgrama_model = require 'create_model_mmskipgrama'
print(mmskipgrama_model)

--------------------------------------------------------------
-- SETTINGS
local opt = { nonlinearity_type = 'requ' }

-- function that numerically checks gradient of the loss:
-- f is the scalar-valued function
-- g returns the true gradient (assumes input to f is a 1d tensor)
-- returns difference, true gradient, and estimated gradient
local function checkgrad(f, g, x, eps)
  -- compute true gradient
  local grad = g(x)
  -- print(grad)
  print('grad size: ', grad:size())
  
  -- compute numeric approximations to gradient
  local eps = eps or 1e-7
  local grad_est = torch.DoubleTensor(grad:size())
  for i = 1, grad:size(1) do
    -- TODO: do something with x[i] and evaluate f twice, and put your estimate of df/dx_i into grad_est[i]
    x[i] = x[i] + eps
    local z, y = f(x)
    x[i] = x[i] - 2.0 * eps
    z2, y = f(x)
    z = z - z2
    grad_est[i] = z / (2.0 * eps)
    x[i] = x[i] + eps
  end

  -- computes (symmetric) relative error of gradient
  local diff = torch.norm(grad - grad_est) / torch.norm(grad + grad_est)
  return diff, grad, grad_est
end

function fakedata(n)
    local data = {}
    local dim = 4
    local num_neg = 20
    local vocab_size = 100 
    local imgFVocab = torch.randn(vocab_size, dim)
    local imgF = torch.randn(n, dim)
    local negF = torch.Tensor(n, dim, num_neg)
    local word_id = torch.Tensor(n);
    word_id:random(vocab_size);
    local posimage_id = word_id:clone()
    for i = 1, n do
        imgF[{{i}, {}}] = imgFVocab[{{posimage_id[i]}, {}}]
        for k = 1, num_neg do
            local negid = torch.random(vocab_size)
            while negid == posimage_id[i] do
                negid = torch.random(vocab_size)
            end
            negF[{{i}, {}, {k}}] = imgFVocab[{{negid}, {}}]
        end
    end
    print(imgF)
    print(negF)
    data.inputs = {word_id, imgF, negF}
    data.targets = torch.rand(n):mul(3):add(1):floor()  -- random integers from {1,2,3}
    print(data)
    print('-------------------------------------------------')
    return data
end

---------------------------------------------------------
-- generate fake data, then do the gradient check
--
torch.manualSeed(1)
-- local data = fakedata(5)
local data = fakedata(500)
local model, criterion = mmskipgrama_model:create_model_mmskipgrama(opt)
local parameters, gradParameters = model:getParameters()
print('88888888888888888888888888888888888888888888888')
-- print(parameters)
print(gradParameters)
print('88888888888888888888888888888888888888888888888')

-- returns loss(params)
local f = function(x)
  if x ~= parameters then
    parameters:copy(x)
  end
  print('Here5')
  return criterion:forward(model:forward(data.inputs), data.targets)
end
-- returns dloss(params)/dparams
local g = function(x)
  if x ~= parameters then
    parameters:copy(x)
  end
  gradParameters:zero()

  print('Here1')
  local outputs = model:forward(data.inputs)
  print('Here2')
  criterion:forward(outputs, data.targets)
  print('Here3')
  -- model:backward(data.inputs, criterion:backward(outputs, data.targets))
  local gradInput = criterion:backward(outputs, data.targets)
  local gradInputModel = model:backward(data.inputs, gradInput)
  print('Here4')
  print(gradInput)
  print(gradInputModel)
  -- print(gradParameters)

  return gradParameters
end

local diff = checkgrad(f, g, parameters)
print(diff)

