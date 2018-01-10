-- PIXEL MSE CRITERION
-- Created by IntelliJ IDEA.
-- User: asebag
-- Date: 12/27/17
-- Time: 4:23 PM
-- To change this template use File | Settings | File Templates.
--
--Checked that the gradient value is approx. equivalent to the differences finies using test.lua
----> it works

local CrispPixelMSE, parent = torch.class('nn.CrispPixelMSE', 'nn.Criterion')

function CrispPixelMSE:__init(power, epsilon)
    parent.__init(self)

    self.power = power or 2
    self.eps = epsilon or 1e-6
end

function CrispPixelMSE:multiplicator(target)

    return torch.add(
                        torch.add(target, self.eps):pow(-self.power),
                        torch.add(-target, 1+self.eps):pow(-self.power)
                     )
end

function CrispPixelMSE:updateOutput(input,target)

    self.output = (input-target):pow(2)
    self.output = (self.output:cmul(self:multiplicator(target))
                                            ):sum()

    self.output = self.output/input:nElement()

    return self.output
end

function CrispPixelMSE:updateGradInput(input, target)
    self.gradInput = (input-target):cmul(self:multiplicator(target)):mul(2/input:nElement())

   return self.gradInput
end


