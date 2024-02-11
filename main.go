package main

import (
	"encoding/json"
	"fmt"
	"image"
	"image/gif"
	"image/jpeg"
	"image/png"
	"io"
	"log"
	"math"
	"net/http"
	"os"
	"runtime/debug"
	"strings"

	tf "github.com/galeone/tensorflow/tensorflow/go"
	tg "github.com/galeone/tfgo"
	"golang.org/x/image/webp"

	"github.com/nfnt/resize"
)

var (
	model *tg.Model
)

type classification struct {
	Drawings float32 `json:"drawings"`
	Hentai   float32 `json:"hentai"`
	Neutral  float32 `json:"neutral"`
	Porn     float32 `json:"porn"`
	Sexy     float32 `json:"sexy"`
}

func main() {
	os.Setenv("TF_CPP_MIN_LOG_LEVEL", "2")

	model = tg.ImportModel("./inception_v3/nsfw.299x299.pb", "", nil)

	log.Println("Run server ....")
	http.HandleFunc("/image", image2Handler)

	err := http.ListenAndServe(":8080", nil)

	if err != nil {
		log.Fatalln(err)
	}
}

func image2Handler(w http.ResponseWriter, r *http.Request) {
	// Read image
	imageFile, header, err := r.FormFile("image")
	if err != nil {
		log.Printf("unable to read image: %v", err)

		w.WriteHeader(http.StatusBadRequest)
		w.Write([]byte("400 - Bad Request"))
		w.Write([]byte(err.Error()))
	}

	defer imageFile.Close()

	print(header.Filename)

	classifications, err := modelExecInception(imageFile, header.Filename)
	if err != nil {
		log.Printf("unable to make a prediction: %v", err)
		w.WriteHeader(http.StatusInternalServerError)
	}

	log.Println(classifications)

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(classifications)
}

func modelExecInception(imageFile io.ReadCloser, filename string) (*classification, error) {
	normalizedImg, err := createTensor(imageFile, filename, 299, 299)
	if err != nil {
		log.Printf("unable to make a normalizedImg from image: %v", err)
		return nil, err
	}

	results := model.Exec(
		[]tf.Output{
			model.Op("dense_3/Softmax", 0),
		}, map[tf.Output]*tf.Tensor{
			model.Op("input_1", 0): normalizedImg,
		},
	)

	probabilities := results[0].Value().([][]float32)[0]

	classifications := classification{
		Drawings: float32(toFixed(float64(probabilities[0]), 4)),
		Hentai:   float32(toFixed(float64(probabilities[1]), 4)),
		Neutral:  float32(toFixed(float64(probabilities[2]), 4)),
		Porn:     float32(toFixed(float64(probabilities[3]), 4)),
		Sexy:     float32(toFixed(float64(probabilities[4]), 4)),
	}

	return &classifications, nil
}

func round(num float64) int {
	return int(num + math.Copysign(0.5, num))
}

func toFixed(num float64, precision int) float64 {
	output := math.Pow(10, float64(precision))
	return float64(round(num*output)) / output
}

func createTensor(src io.ReadCloser, fileName string, imageHeight, imageWidth int) (*tf.Tensor, error) {
	var srcImage image.Image
	var err error

	split := strings.Split(fileName, ".")
	ext := strings.ToLower(split[len(split)-1])
	switch ext {
	case "png":
		srcImage, err = png.Decode(src)
	case "jpg", "jpeg":
		srcImage, err = jpeg.Decode(src)
	case "gif":
		srcImage, err = gif.Decode(src)
	case "webp":
		srcImage, err = webp.Decode(src)

	default:
		return nil, fmt.Errorf("unsupported image extension %s", ext)
	}

	if err != nil {
		return nil, err
	}

	img := resize.Resize(uint(imageWidth), uint(imageHeight), srcImage, resize.Bilinear)

	return imageToTensor(img, imageHeight, imageWidth)
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
