-- MMD CRITERION
-- Created by: Alice Schoenauer Sebag
-- Date: 22/10/17
-- Torch version of MMD criterion. Two improvements with 'raw' MMD [Gretton et al., 2012]
    -- a. Approximating each kernel embeddings by low-dimension projections according to Bochner thm as described
        --in [Lopez-Paz et al., 2015] sec. 3.3, 
    -- b. Using multiple kernel bandwidths to alleviate the dependency on sample size, as described
        --in [Goudet et al., 2017] sec 4.1
-- Tested using nn.test.criterionJacobianTest, gradients should be correct

local ApproxMMDCriterion, parent = torch.class('nn.ApproxMMDCriterion', 'nn.Criterion')

function ApproxMMDCriterion:__init(input_dimension, num_basis_vectors, domain_batchsize)
    parent.__init(self)

    local domain_batchsize = domain_batchsize or 64
    --Attributes specific for approx MMD criterion
    self.input_dimension  = input_dimension or 256
        --All the bandwidths which will be used so that final kernel = sum_gamma rbf kernel(gamma)
    self.gammas = torch.Tensor({0.005, 0.05, 0.25, 0.5, 1, 5, 50}):mul(self.input_dimension)
        --Pick nb of basis vectors to approximate kernel
    self.num_basis_vectors = num_basis_vectors or 100 
    self.ngamma = self.gammas:size(1)
    self.buffer = torch.Tensor(self.ngamma, self.num_basis_vectors, 2*domain_batchsize)
    self.gradBuffer = torch.Tensor(self.num_basis_vectors)

        --Sampling the angles and bias vectors to approximate each kernel
    self.thetas = torch.randn(self.ngamma, self.num_basis_vectors, self.input_dimension)
    for i,gamma in ipairs(self.gammas:totable()) do
        self.thetas[i]:mul(gamma)
    end
    self.B = torch.rand(self.ngamma, self.num_basis_vectors):mul(2*math.pi)
   
end

function ApproxMMDCriterion:product(basis_vector_ind, vec)
    return torch.addmv(self.B[{{}, { basis_vector_ind }}]:resize(self.ngamma),
                        self.thetas[{{},{ basis_vector_ind },{}}]:resize(self.ngamma, self.input_dimension),
                        vec:resize(self.input_dimension))
end

function ApproxMMDCriterion:sum_cos_products(batchsize, basis_vector_ind, input_index)
    return torch.sum(
                    torch.cos(self.buffer[{{},{basis_vector_ind},{input_index}}])
                    - torch.cos(self.buffer[{{},{basis_vector_ind},{batchsize+input_index}}])
    )
end

function ApproxMMDCriterion:updateOutput(input,y)
    local batchsize = input:size(1)/2
    self.output = 0
    self.buffer:zero()
    self.gradBuffer:zero()

    for j=1,self.num_basis_vectors do
        for i=1, batchsize do
            self.buffer[{{},{j},{i}}] = self:product(j, input[i])
            self.buffer[{{},{j},{batchsize+i}}] = self:product(j, input[batchsize+i])

            self.gradBuffer[j] = self.gradBuffer[j] + self:sum_cos_products(batchsize, j, i)
        end
        self.output = self.output + self.gradBuffer[j]^2
    end
    
    self.output = self.output*4/(self.num_basis_vectors^2 * batchsize^2 * self.ngamma^2)
    return self.output
end

function ApproxMMDCriterion:updateGradInput(input, y)
    local batchsize = input:size(1)/2
    self.gradInput:resizeAs(input)

    for i =1,batchsize do
        for p=1,self.input_dimension do
            for j=1,self.num_basis_vectors do
                local curr_vec = self.thetas[{{},{j},{p}}]
                self.gradInput[{{ i }, {p}}] = self.gradInput[{{ i }, {p}}] -self.gradBuffer[j]* torch.dot(curr_vec, torch.sin(self.buffer[{{},{j},{ i }}]))

                self.gradInput[{{batchsize + i }, {p}}] = self.gradInput[{{batchsize + i }, {p}}] + self.gradBuffer[j] * torch.dot(curr_vec, torch.sin(self.buffer[{{},{j},{batchsize + i }}]))
            end
        end
    end

   self.gradInput = self.gradInput * 8 /(self.num_basis_vectors^2 * batchsize^2 * self.ngamma^2)
   return self.gradInput
end


