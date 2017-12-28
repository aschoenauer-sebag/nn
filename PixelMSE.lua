-- PIXEL MSE CRITERION
-- Created by IntelliJ IDEA.
-- User: asebag
-- Date: 12/27/17
-- Time: 4:23 PM
-- To change this template use File | Settings | File Templates.
--
--Checked that the gradient value is approx. equivalent to the differences finies using test.lua
----> it works

local PixelMSE, parent = torch.class('nn.PixelMSE', 'nn.Criterion')

function PixelMSE:__init(weights, threshold)
    parent.__init(self)

    self.num_bins = weights:size(1)
    self.weights = weights

    self.threshold = threshold
end

function PixelMSE:compute_VdF_matrix(target)
    local a,b,c,d = unpack(torch.totable(target:size()))
    local V = torch.cat(
                target:lt(self.threshold), target:ge(self.threshold), 5
    )
    V=V:reshape(a*b*c*d, self.num_bins)
    self.VdF = torch.mv(V:cuda(), self.weights):reshape(a,b,c,d)

end

function PixelMSE:updateOutput(input,target)
    self:compute_VdF_matrix(target)
    --of shape mbx 2 x w x w
    self.output = (input-target):pow(2)
    self.output = (self.output:cmul(self.VdF)):sum()
    self.output = self.output/input:nElement()

    return self.output
end

function PixelMSE:updateGradInput(input, target)
    self.gradInput = (input-target):cmul(self.VdF):mul(-2/input:nElement())
   return self.gradInput
end


