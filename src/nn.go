package core

type neuron struct {
	v       *value
	weights []*value
	bias    *value
}

type mlp struct {
	input    layer
	hidden   []layer
	output   layer
	expected []*value
	Loss     *value
}

type layer []*neuron

func NewNeuron(prevLayer layer) *neuron {
	neuron := &neuron{
		weights: RandValues(len(prevLayer)),
		bias:    Value(seededRand.Float64()),
	}

	// iterate over all weights and input and multiply them (prev + w_i * x_i)
	if len(prevLayer) > 0 {
		prevTerm := prevLayer[0].v.Mult(neuron.weights[0])
		for i := 1; i < len(neuron.weights); i++ {
			prevTerm = prevTerm.Add(prevLayer[i].v.Mult(neuron.weights[i]))
		}

		// + bias
		neuron.v = prevTerm.Add(neuron.bias).ReLU()
	} else {
		// else it is an input neuron
		neuron.v = Value(0)
	}

	return neuron
}

func NewLayer(size int, prevLayer layer) layer {
	if size < 1 {
		panic("You need at least one neuron in a layer(Err: size<1)")
	}

	l := []*neuron{}

	for i := 0; i < size; i++ {
		l = append(l, NewNeuron(prevLayer))
	}

	return l
}

func NewMLP(nInputs int, hidden []int, nOutputs int) *mlp {
	if nInputs < 1 {
		panic("You need at least one input neuron(Err: nin<1)")
	}

	if nOutputs < 1 {
		panic("You need at least one output neuron(Err: nout<1)")
	}

	inputLayer := NewLayer(nInputs, layer{})

	prevOutput := inputLayer
	h := mapSlice(hidden, func(size int, prev int) layer {
		currentHiddenLayer := NewLayer(size, prevOutput)
		prevOutput = currentHiddenLayer
		return currentHiddenLayer
	})

	outputLayer := NewLayer(nOutputs, prevOutput)
	expected := InitValues(nOutputs, 0)

	loss := expected[0].Sub(outputLayer[0].v).Pow(2)
	for i := 1; i < len(expected); i++ {
		loss = loss.Add(expected[i].Sub(outputLayer[i].v).Pow(2))
	}

	mlp := &mlp{
		input:    inputLayer,
		hidden:   h,
		output:   outputLayer,
		expected: expected,
		Loss:     loss,
	}

	return mlp
}

func (mlp *mlp) SetInput(inputs []float64) {
	if len(inputs) != len(mlp.input) {
		panic("the number of input needs to match the size of the input layer(Err: len(inputs)!=len(mlp.input))")
	}

	for i, n := range mlp.input {
		n.v.SetData(inputs[i])
	}
}

func (mlp *mlp) SetExpected(expected []float64) {
	for i, v := range mlp.expected {
		v.SetData(expected[i])
	}
}

func (mlp *mlp) GetParameters() []*value {
	params := []*value{}
	// weights and biases of the hidden layers
	for _, l := range mlp.hidden {
		for _, n := range l {
			params = append(params, n.weights...)
			params = append(params, n.bias)
		}
	}

	// weights and biases of the output layer
	for _, n := range mlp.output {
		params = append(params, n.weights...)
		params = append(params, n.bias)
	}

	return params
}

func (mlp *mlp) TuneParameters(rate float64) {
	params := mlp.GetParameters()
	for _, p := range params {
		p.data -= p.grad * rate
	}
}
