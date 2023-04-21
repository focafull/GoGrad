package core

import (
	"fmt"
	"log"

	"github.com/goccy/go-graphviz"
	"github.com/goccy/go-graphviz/cgraph"
)

func (v *value) Sprint() string {
	return fmt.Sprintf("d: %.3f, g: %.3f\n", v.data, v.grad)
}

func Visualize(v *value, path string) {
	g := graphviz.New()
	graph, err := g.Graph()
	handleErr(err)

	var buildGraph func(self *value, prev *cgraph.Node)
	buildGraph = func(s *value, n *cgraph.Node) {
		// create root node
		number, err := graph.CreateNode(fmt.Sprintf("%p", s))
		handleErr(err)
		number.SetLabel(s.Sprint())

		// create edge if this node is not the end node
		if n != nil {
			_, err = graph.CreateEdge("e", number, n)
			handleErr(err)
		}

		prev := number
		if s.operation != "" {
			// create operations node if we have an operation
			prev, err = graph.CreateNode(fmt.Sprintf("%p_o", s))
			handleErr(err)
			prev.SetLabel(fmt.Sprint(s.operation))

			_, err = graph.CreateEdge("e", prev, number)
			handleErr(err)

			prev.SetFontColor("red")
			number.SetFontColor("green")
		} else {
			// else this is a leaf node
			number.SetFontColor("blue")
		}

		// traverse the children
		for _, v := range s.children {
			buildGraph(v, prev)
		}
	}

	buildGraph(v, nil)

	err = g.RenderFilename(graph, graphviz.PNG, path)
	handleErr(err)
}

func handleErr(err error) {
	if err != nil {
		log.Fatal(err)
	}
}
