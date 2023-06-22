package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"image"
	"image/jpeg"
	"io"
	"log"
	"net/http"
	"os"
	"runtime/debug"
	"strings"

	"github.com/disintegration/imaging"

	tf "github.com/galeone/tensorflow/tensorflow/go"
	tg "github.com/galeone/tfgo"
)

var (
	model  *tg.Model
	labels []string
)

type classification struct {
	Label       string  `json:"label"`
	Probability float32 `json:"probability"`
}

func main() {
	os.Setenv("TF_CPP_MIN_LOG_LEVEL", "2")

	modelName := "./mobilenet_v2_140_224"
	loadLabels(modelName)
	model = tg.LoadModel(modelName, []string{"serve"}, nil)

	log.Println("Run server ....")
	http.HandleFunc("/image", imageHandler)

	err := http.ListenAndServe(":8080", nil)

	if err != nil {
		log.Fatalln(err)
	}
}

func imageHandler(w http.ResponseWriter, r *http.Request) {
	// Read image
	imageFile, header, err := r.FormFile("image")
	if err != nil {
		log.Fatalf("unable to read image: %v", err)
	}
	defer imageFile.Close()

	print(header.Filename)

	normalizedImg, err := createTensor(imageFile)
	if err != nil {
		log.Fatalf("unable to make a normalizedImg from image: %v", err)
	}

	results := model.Exec(
		[]tf.Output{
			model.Op("StatefulPartitionedCall", 0),
		}, map[tf.Output]*tf.Tensor{
			model.Op("serving_default_input", 0): normalizedImg,
		},
	)

	probabilities := results[0].Value().([][]float32)[0]
	classifications := []classification{}
	for i, p := range probabilities {
		classifications = append(classifications, classification{
			Label:       strings.ToLower(labels[i]),
			Probability: p,
		})
		labelText := strings.ToLower(labels[i])
		fmt.Printf("%s %f \n", labelText, p)
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(classifications)
}

func loadLabels(path string) error {
	modelLabels := path + "/labels.txt"
	f, err := os.Open(modelLabels)
	if err != nil {
		return err
	}
	defer f.Close()
	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		labels = append(labels, scanner.Text())
	}

	if err := scanner.Err(); err != nil {
		return err
	}

	return nil
}

func createTensor(image io.ReadCloser) (*tf.Tensor, error) {
	srcImage, err := jpeg.Decode(image)
	if err != nil {
		log.Fatalf("unable to decode image: %v", err)
	}
	img := imaging.Fill(srcImage, 224, 224, imaging.Center, imaging.Lanczos)
	return imageToTensor(img, 224, 224)
}

func imageToTensor(img image.Image, imageHeight, imageWidth int) (tfTensor *tf.Tensor, err error) {
	defer func() {
		if r := recover(); r != nil {
			err = fmt.Errorf("classify: %s (panic)\nstack: %s", r, debug.Stack())
		}
	}()

	if imageHeight <= 0 || imageWidth <= 0 {
		return tfTensor, fmt.Errorf("classify: image width and height must be > 0")
	}

	var tfImage [1][][][3]float32

	for j := 0; j < imageHeight; j++ {
		tfImage[0] = append(tfImage[0], make([][3]float32, imageWidth))
	}
	for i := 0; i < imageWidth; i++ {
		for j := 0; j < imageHeight; j++ {
			r, g, b, _ := img.At(i, j).RGBA()
			tfImage[0][j][i][0] = convertValue(r)
			tfImage[0][j][i][1] = convertValue(g)
			tfImage[0][j][i][2] = convertValue(b)
		}
	}
	return tf.NewTensor(tfImage)
}

func convertValue(value uint32) float32 {
	return (float32(value >> 8)) / float32(255)
}
