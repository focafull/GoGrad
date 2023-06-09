package main

import (
	"fmt"
	"math/rand"

	. "focafull.de/goGrad/src"
)

func main() {
	mlp := NewMLP(2, []int{16, 16}, 1)

	// trainingsData := map[float64][]float64{}
	// trainingsData[0] = []float64{1, 0}
	// trainingsData[1] = []float64{0, 1}
	// trainingsData[2] = []float64{1, 0}
	// trainingsData[3] = []float64{0, 1}
	// trainingsData[4] = []float64{1, 0}
	// trainingsData[5] = []float64{0, 1}
	// trainingsData[6] = []float64{1, 0}
	// trainingsData[7] = []float64{0, 1}
	// trainingsData[8] = []float64{1, 0}
	// trainingsData[9] = []float64{0, 1}
	// trainingsData[10] = []float64{1, 0}
	// trainingsData[11] = []float64{0, 1}
	// trainingsData[12] = []float64{1, 0}
	// trainingsData[13] = []float64{0, 1}
	// trainingsData[14] = []float64{1, 0}
	// trainingsData[15] = []float64{0, 1}
	// trainingsData[16] = []float64{1, 0}
	// trainingsData[17] = []float64{0, 1}
	// trainingsData[18] = []float64{1, 0}
	// trainingsData[19] = []float64{0, 1}

	// Visualize(mlp.Loss, fmt.Sprintf("out/mlp.png"))
	for i := 0; i < 10; i++ {

		mlp.Loss.Zero_grad()
		mlp.Loss.Backward()
		mlp.TuneParameters(0.0000000000000001)

		x := rand.Float64() * 1000
		y := rand.Float64() * 1000
		e := 1.
		if y > x {
			e = 0.
		}

		mlp.SetInput([]float64{x, y})
		mlp.SetExpected([]float64{e})
		mlp.Loss.Forward()

		Visualize(mlp.Loss, fmt.Sprintf("out/mlp%d.png", i))

		if i%1000 == 0 {
			fmt.Printf("%f\n", mlp.Loss.GetData())
		}
	}
}
