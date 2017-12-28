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

function PixelMSE:__init(bin_inverse_frequencies, bin_width)
    parent.__init(self)

    self.bin_width = bin_width

    self.num_bins = bin_inverse_frequencies:size(1)
    self.inv_fqcies = bin_inverse_frequencies

end

function PixelMSE:compute_VdF_matrix(target)
    local a,b,c,d = unpack(torch.totable(target:size()))
    local V = torch.zeros(a,b,c,d, self.num_bins)
    for i=1, self.num_bins do
        V[{{},{},{},{},{i}}] = (target:ge(self.bin_width*(i-1)) + target:lt(self.bin_width*i)):eq(2)
    end
    V=V:reshape(a*b*c*d, self.num_bins)
    self.VdF = torch.mv(V, self.inv_fqcies):reshape(a,b,c,d)

end

function PixelMSE:updateOutput(input,target)
    self:compute_VdF_matrix(target)
    --of shape mbx 2 x w x w
    self.sqrtL = input-target

    self.output = self.sqrtL:clone():pow(2)
    self.output = (self.output:cmul(self.VdF)):sum()
    self.output = self.output/input:nElement()

    return self.output
end

function PixelMSE:updateGradInput(input, target)
    self.gradInput = self.sqrtL:cmul(self.VdF):mul(-2/input:nElement())
   return self.gradInput
end


