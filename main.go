package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"image"
	"image/gif"
	"image/jpeg"
	"image/png"
	"io"
	"log"
	"net/http"
	"os"
	"os/exec"
	"runtime/debug"
	"strings"

	"github.com/disintegration/imaging"
	uuid "github.com/google/uuid"
	"golang.org/x/image/webp"

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
	Max         float32 `json:"max,omitempty"`
}

func main() {
	os.Setenv("TF_CPP_MIN_LOG_LEVEL", "2")

	modelName := "./mobilenet_v2_140_224"
	loadLabels(modelName)
	model = tg.LoadModel(modelName, []string{"serve"}, nil)

	log.Println("Run server ....")
	http.HandleFunc("/image", imageHandler)
	http.HandleFunc("/video", videoHandler)

	err := http.ListenAndServe(":8080", nil)

	if err != nil {
		log.Fatalln(err)
	}
}

func videoHandler(w http.ResponseWriter, r *http.Request) {
	videoFile, header, err := r.FormFile("video")
	if err != nil {
		log.Printf("unable to read video: %v", err)

		w.WriteHeader(http.StatusBadRequest)
		w.Write([]byte("400 - Bad Request"))
		w.Write([]byte(err.Error()))
	}

	defer videoFile.Close()

	print(header.Filename)

	split := strings.Split(header.Filename, ".")
	ext := strings.ToLower(split[len(split)-1])

	uid := uuid.New().String()

	filename := uid + "." + ext
	dst, err := os.Create(filename)
	if err != nil {
		log.Println("error creating file", err)
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	defer dst.Close()
	if _, err := io.Copy(dst, videoFile); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	classifications, err := classifyVideo(filename, uid)

	if err != nil {
		w.WriteHeader(http.StatusInternalServerError)
	}

	for _, c := range classifications {
		log.Printf("%s (%0.2f%%)\n", c.Label, c.Probability*100)
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(classifications)
}

func classifyVideo(filename string, uid string) ([]classification, error) {
	extractFrames(filename, uid)

	files, err := os.ReadDir(uid)
	if err != nil {
		log.Printf("unable to read images: %v", err)

		return nil, err
	}

	classifications := [][]classification{}

	for _, file := range files {
		if !file.IsDir() {
			f, err := os.Open(uid + "/" + file.Name())
			if err != nil {
				log.Printf("unable to read image: %v", err)

				return nil, err
			}

			defer f.Close()

			classification, err := modelExec(f, file.Name())
			if err != nil {
				log.Printf("unable to make a prediction: %v", err)

				return nil, err
			}

			classifications = append(classifications, classification)
		}
	}

	os.RemoveAll(uid)
	avg := map[string]float32{}
	max := map[string]float32{}

	for _, r := range classifications {
		for _, c := range r {
			avg[c.Label] += c.Probability

			if max[c.Label] < c.Probability {
				max[c.Label] = c.Probability
			}
		}
	}

	result := []classification{}
	for k, v := range avg {
		result = append(result, classification{
			Label:       k,
			Probability: v / float32(len(classifications)),
			Max:         max[k],
		})
	}
	return result, nil
}

func imageHandler(w http.ResponseWriter, r *http.Request) {
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

	classifications, err := modelExec(imageFile, header.Filename)
	if err != nil {
		log.Printf("unable to make a prediction: %v", err)
		w.WriteHeader(http.StatusInternalServerError)
	}

	for _, c := range classifications {
		log.Printf("%s (%0.2f%%)\n", c.Label, c.Probability*100)
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(classifications)
}

func modelExec(imageFile io.ReadCloser, filename string) ([]classification, error) {
	normalizedImg, err := createTensor(imageFile, filename)
	if err != nil {
		log.Printf("unable to make a normalizedImg from image: %v", err)
		return nil, err
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
	}

	return classifications, nil
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

func createTensor(src io.ReadCloser, fileName string) (*tf.Tensor, error) {
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

func extractFrames(src string, dst string) error {
	err := os.Mkdir(dst, 0755)
	if err != nil {
		log.Println("error creating folder", err)

		return err
	}

	cmdArgs := []string{}
	cmdArgs = append(cmdArgs, "-i")
	cmdArgs = append(cmdArgs, src)
	cmdArgs = append(cmdArgs, "-vf")
	cmdArgs = append(cmdArgs, "fps=1")
	cmdArgs = append(cmdArgs, dst+"/frame_%d.jpg")

	log.Printf("ffmpeg %s", cmdArgs)
	cmd := exec.Command("ffmpeg", cmdArgs...)

	out, err := cmd.CombinedOutput()

	if err != nil {
		return err
	}

	printOutput(out)

	return nil
}

func printOutput(out []byte) {
	if len(out) > 0 {
		log.Printf("Output: %s\n", string(out))
	}
}
