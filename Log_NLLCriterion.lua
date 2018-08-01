--
-- Created by IntelliJ IDEA.
-- User: asebag
-- Date: 7/30/18
-- Time: 12:18 PM
-- To change this template use File | Settings | File Templates.
--

local Log_NLLCriterion, Criterion = torch.class('nn.Log_NLLCriterion', 'nn.Criterion')

function Log_NLLCriterion:__init(weights, sizeAverage)
   Criterion.__init(self)
   self.log = nn.Log()
   self.negloglik_criterion = nn.ClassNLLCriterion(weights, sizeAverage)
   self.sizeAverage = self.negloglik_criterion.sizeAverage
end

function Log_NLLCriterion:updateOutput(input, target)
   local out = self.log:forward(input)
   self.output = self.negloglik_criterion:forward(out, target)

   return self.output
end

function Log_NLLCriterion:updateGradInput(input, target)
   local criDf = self.negloglik_criterion:backward(self.log.output, target)
   self.gradInput = self.log:backward(input, criDf)

   return self.gradInput
end

return nn.Log_NLLCriterion
