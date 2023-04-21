package core

import (
	"testing"

	. "github.com/onsi/gomega"
)

func setup(t *testing.T) {
	RegisterFailHandler(func(message string, callerSkip ...int) {
		t.Log(message)
		t.Fail()
	})
}

func TestNewNeuron(t *testing.T) {
	setup(t)
	n1 := NewNeuron(layer{})
	n2 := NewNeuron(layer{})

	Expect(len(n1.weights)).To(Equal(0))
	Expect(len(NewNeuron(layer{n1, n2}).weights)).To(Equal(2))
}

func TestNewLayer(t *testing.T) {
	setup(t)
	l1 := NewLayer(3, layer{})
	Expect(len(l1)).To(Equal(3))
	Expect(len(l1[0].weights)).To(Equal(0))
	Expect(len(l1[1].weights)).To(Equal(0))
	Expect(len(l1[2].weights)).To(Equal(0))

	l2 := NewLayer(2, l1)
	Expect(len(l2)).To(Equal(2))
	Expect(len(l2[0].weights)).To(Equal(3))
	Expect(len(l2[1].weights)).To(Equal(3))
}

func TestNewMLP(t *testing.T) {
	setup(t)
	mlp := NewMLP(2, []int{4, 5}, 3)
	Expect(len(mlp.input)).To(Equal(2))
	Expect(len(mlp.input[0].weights)).To(Equal(0))

	Expect(len(mlp.hidden[0])).To(Equal(4))
	Expect(len(mlp.hidden[0][0].weights)).To(Equal(2))

	Expect(len(mlp.hidden[1])).To(Equal(5))
	Expect(len(mlp.hidden[1][3].weights)).To(Equal(4))

	Expect(len(mlp.output)).To(Equal(3))
	Expect(len(mlp.output[0].weights)).To(Equal(5))
}

func TestParameters(t *testing.T) {
	setup(t)

	mlp := NewMLP(2, []int{4, 5}, 3)
	params := mlp.GetParameters()

	Expect(len(params)).To(Equal((2*4 + 4) + (4*5 + 5) + (5*3 + 3)))
}
