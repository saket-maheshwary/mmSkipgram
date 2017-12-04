require 'nn'

local ReQU = torch.class('nn.ReQU', 'nn.Module')

function ReQU:updateOutput(input)
  self.output:resizeAs(input):copy(input)
  self.output = torch.gt(self.output, torch.Tensor(self.output:size()):zero()):double():cmul(self.output):pow(2)
  return self.output
end

function ReQU:updateGradInput(input, gradOutput)
  self.gradInput:resizeAs(gradOutput):copy(gradOutput)
  self.gradInput = torch.gt(input, torch.Tensor(input:size()):zero()):double():cmul(input):mul(2):cmul(gradOutput)
  return self.gradInput
end

