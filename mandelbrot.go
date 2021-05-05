package main

import (
	"flag"
	"fmt"
	"image"
	"image/color"
	"image/png"
	"math"
	"os"
	"runtime"
	"sync"
	"time"
)

//Written by J Sörngård 2020.
func main() {

	beginning := time.Now() //In the beginning there was only the vacuum tube...

	//Establish and parse command line arguments.
	verboseFlag := flag.Bool("q", false, "use this flag to run the program silently")
	aspectRatioFlag := flag.Float64("r", 16.0/9.0, "set the aspect ratio of the image")
	realFractalCenterFlag := flag.Float64("cr", -0.75, "set the real part of the center of the fractal image")
	imagFractalCenterFlag := flag.Float64("ci", 0, "set the imaginary part of the center of the fractal image")
	resolutionFlag := flag.Int("y", 2160, "set the number of pixels along the y-axis")
	ssaaFlag := flag.Int("s", 3, "set the number of supersamples along one direction. Execution speed slows down with the square of this number")
	zoomFlag := flag.Float64("z", 1, "the factor with witch to zoom the image")
	noWriteImageFlag := flag.Bool("x", false, "use this flag to stop the program from writing the image to disk")
	gammaFlag := flag.Float64("g", 1.0, "set the gamma factor of the image (not actually gamma, but works like it for values between 0.88 and 3)")
	flag.Parse()

	//We use this flag to determine whether to print extra information.
	verbose := !*verboseFlag

	//This is set to the maximum of an unsigned 8-bit integer
	//since that is the scale of RGB values in images.
	maxIterations := int(^uint8(0))

	//The name of the final image.
	outputName := "m.png"

	//How much to zoom the image.
	zoom := *zoomFlag

	//Will raise the result of the IteratePoint function to
	//this number.
	gamma := *gammaFlag

	//Whether to write the resulting image to disk.
	writeImage := !*noWriteImageFlag

	//Work out size of plot in complex plane.
	aspectRatio := *aspectRatioFlag
	imagDistance := 8.0 / 3.0 / zoom //The distance covered along the imaginary axis in the image.
	realDistance := imagDistance * aspectRatio

	//Retrieve image center.
	realFractalCenter := *realFractalCenterFlag
	imagFractalCenter := -*imagFractalCenterFlag //The minus sign counters some sign error I can not find.

	//Set start and end points.
	realStart := realFractalCenter - realDistance/2.0
	realEnd := realFractalCenter + realDistance/2.0
	imagStart := imagFractalCenter - imagDistance/2.0
	imagEnd := imagFractalCenter + imagDistance/2.0

	//Get supersampling factor
	ssaa := *ssaaFlag

	//Determine the number of points along each axis.
	imagPointsLen := *resolutionFlag
	realPointsLen := int(float64(imagPointsLen) * aspectRatio)

	//Initialize the slices that hold the real and imaginary parts.
	realPoints := make([]float64, realPointsLen)
	imagPoints := make([]float64, imagPointsLen)

	if verbose {
		fmt.Printf("------ Generating %d by %d image ------\n", realPointsLen, imagPointsLen)
	}

	//Make an image object.
	upLeft := image.Point{0, 0}
	downRight := image.Point{realPointsLen, imagPointsLen}
	img := image.NewRGBA(image.Rectangle{upLeft, downRight})

	//Work out position of points in complex plane.
	realPoints = LinearSpace(realStart, realEnd, realPointsLen)
	realDelta := (realEnd - realStart) / float64(realPointsLen-1)
	imagPoints = LinearSpace(imagStart, imagEnd, imagPointsLen)
	imagDelta := (imagEnd - imagStart) / float64(imagPointsLen-1)

	//Work out if we should mirror the fractal image from the top,
	//bottom, or not at all.
	mirror := imagStart < 0 && imagEnd > 0 //We mirror if the image contains both the positive and negative imaginary axis.
	bottomHalfLarger := math.Abs(imagStart) >= math.Abs(imagEnd)
	mirrorBottom := mirror && bottomHalfLarger //We mirror from the larger part to the smaller part.
	mirrorTop := mirror && !bottomHalfLarger

	displayIterationProgress := verbose && imagPointsLen >= 1080 //For images smaller than these sizes the progress is pretty much instant.
	displayMirrorProgress := verbose && imagPointsLen >= 6480

	var iterationProgressHead string
	if verbose {
		if ssaa != 1 {
			iterationProgressHead = fmt.Sprintf("Rendering with SSAAx%d: ", int(math.Pow(float64(ssaa), 2)))
		} else {
			iterationProgressHead = "Rendering: "
		}
	}

	start := time.Now()

	//Establish a queue of jobs and fill it.
	rowJobs := make(chan int, imagPointsLen)
	lastColoredRowIndex := 0

	if mirrorTop {
		imagPoints = ReverseFloat64(imagPoints)
	}
	//In the case where we mirror from the bottom or do not
	//mirror at all we do not need to change the order of the points.

	for i, cImag := range imagPoints {
		rowJobs <- i

		//If we are mirroring the image we do not need to color the
		//entire thing. Only until we cross the real axis.
		if (mirrorBottom && cImag > 0) || (mirrorTop && cImag < 0) {
			lastColoredRowIndex = i
			break
		}
	}
	numJobs := len(rowJobs)
	close(rowJobs) //We're done sending jobs so we close the channel.

	//Parallel execution of cores separate threads that each compute
	//rows of the mandelbrot image taken from the rowJobs channel.
	cores := runtime.NumCPU()
	var iterationWaitGroup sync.WaitGroup
	for i := 0; i < cores; i++ {
		iterationWaitGroup.Add(1) //Add one goroutine to the wait group
		go func() {
			rowWorker(img, rowJobs, realPoints, imagPoints, maxIterations, ssaa, realDelta, imagDelta, gamma)
			iterationWaitGroup.Done()
		}() //make a gorountine that calls a worker, and notifies
		//the work group when it finishes.
	}

	//Display iteration progress.
	if displayIterationProgress {
		displayProgress(rowJobs, numJobs, iterationProgressHead)
	} else if verbose {
		fmt.Println(iterationProgressHead[:len(iterationProgressHead)-2] + "...")
	}

	iterationWaitGroup.Wait() //Await the completion of the iterations.

	if mirror {
		mirrorJobs := make(chan int, realPointsLen)

		//Put the index of every column into the jobs channel.
		for i := 0; i < realPointsLen; i++ {
			mirrorJobs <- i
		}

		//Parallel execution of cores separate threads that each mirror
		//columns specified in the jobs channel.
		var mirrorWaitGroup sync.WaitGroup
		for i := 0; i < cores; i++ {
			mirrorWaitGroup.Add(1)
			go func() {
				mirrorColumnWorker(img, mirrorJobs, lastColoredRowIndex, imagPointsLen, mirrorTop)
				mirrorWaitGroup.Done()
			}()
		}
		close(mirrorJobs) //We're done sending jobs so we close the channel.

		//Display mirroring progress
		if displayMirrorProgress {
			displayProgress(mirrorJobs, realPointsLen, " mirroring: ")
		} else if verbose {
			fmt.Println(" mirroring...")
		}

		mirrorWaitGroup.Wait() //Await the completion of the mirroring operation.
	}

	if verbose {
		fmt.Printf(" took %s.\n", time.Since(start))
	}

	if writeImage {

		if verbose {
			fmt.Printf("Writing %s...\n", outputName)
		}

		//Encode image
		start = time.Now()
		f, err := os.Create(outputName)

		if err != nil {
			panic(fmt.Sprintf("unable to create output image %s\n", outputName))
		}

		png.Encode(f, img)

		if verbose {
			fi, err := os.Stat(outputName)
			if err != nil {
				fmt.Printf(" could not find generated image (looked for %s).\n", outputName)
			} else {
				fmt.Printf(" generated %s with a size of %s.\n", outputName, ByteCountBinary(fi.Size()))
			}

			fmt.Printf(" took %s.\n", time.Since(start))
			fmt.Printf("Total time consumption was %s\n", time.Since(beginning))
		}
	}
}

//LinearSpace returns a slice of length elements such that
//the first element is start, the last is end, and the others
//change in a linear fashon. An attempt at numpys linspace.
func LinearSpace(start float64, end float64, length int) []float64 {
	output := make([]float64, length)
	stepSize := (end - start) / float64(length-1)
	for i := 0; i < length; i++ {
		output[i] = start + float64(i)*stepSize
	}

	return output
}

//ReverseFloat64 returns a new slice with all the elements
//of the input slice in reverse order.
func ReverseFloat64(slice []float64) []float64 {

	output := make([]float64, len(slice))

	for i := 0; i < len(slice); i++ {
		output[i] = slice[len(slice)-1-i]
	}

	return output
}

//ByteCountBinary takes in a number of bytes and returns
//it shortened with the appropriate binary prefix (ki, Mi, Gi).
func ByteCountBinary(b int64) string {
	const unit = 1024
	if b < unit {
		return fmt.Sprintf("%d B", b)
	}
	div, exp := int64(unit), 0
	for n := b / unit; n >= unit; n /= unit {
		div *= unit
		exp++
	}
	return fmt.Sprintf("%.1f %ciB", float64(b)/float64(div), "KMGTPE"[exp])
}

func rowWorker(img *image.RGBA, rowJobs <-chan int, cReal []float64, cImag []float64, maxIterations int, ssaa int, realDelta float64, imagDelta float64, gamma float64) {
	if img.Bounds().Dx() != len(cReal) {
		panic(fmt.Sprintf("rowWorker: width of image must be the same as the length of cReal, but they are %d and %d", img.Bounds().Dx(), len(cReal)))
	}
	if img.Bounds().Dy() != len(cImag) {
		panic(fmt.Sprintf("rowWorker: height of image must be the same as the length of cImag, but they are %d and %d", img.Bounds().Dy(), len(cImag)))
	}

	var escapeSpeed float64
	var columnOffset float64
	var rowOffset float64
	var total float64
	var samplesTaken float64
	var inverseFactor float64
	black := color.RGBA{uint8(0), uint8(0), uint8(0), 0xff}

	if ssaa == 1 {
		inverseFactor = 0.0
	} else {
		inverseFactor = 1.0 / float64(ssaa)
	}
	ssaaFactor := math.Pow(float64(ssaa), 2)

	for rowIndex := range rowJobs {

		for j := 0; j < img.Bounds().Dx(); j++ {

			//This loop does supersampling in a grid pattern for each pixel.
			//and averages the results together.
			total = 0
			samplesTaken = 0.0
			for k := 1; k <= int(ssaaFactor); k++ {
				//Computes offsets. These should range from -1/ssaa
				//to 1/ssaa with a 0 included if ssaa is odd.
				columnOffset = (float64(k%ssaa) - 1.0) * inverseFactor
				rowOffset = (float64(k-1)/float64(ssaa) - 1.0) * inverseFactor
				escapeSpeed = IteratePoint(cReal[j]+rowOffset*realDelta, cImag[rowIndex]+columnOffset*imagDelta, maxIterations)
				total = total + escapeSpeed
				samplesTaken++

				//If we are far away from the fractal and escape quickly we
				//do not need the additional smoothness afforded by supersampling.
				if escapeSpeed > 0.9 {
					break
				}
			}

			//total = total / ssaaFactor
			total = total / samplesTaken

			//Adjust gamma of image.
			//                 Don't need to change the gamma of pure black.
			if gamma != 1.0 && total != 0.0 {
				total = math.Pow(total, gamma)
			}

			if total == 0.0 {
				img.Set(j, rowIndex, black)
			} else {
				//The color curves here have been found through a
				//more or less artistic process, feel free to
				//change them to something else if you prefer.
				img.Set(j, rowIndex, color.RGBA{
					uint8(total * math.Pow(float64(maxIterations), 1.0-math.Pow(total, 45.0)*2.0)),
					uint8(total*70.0 - 880.0*math.Pow(total, 18.0) + 701.0*math.Pow(total, 9.0)),
					uint8(total*80.0 + math.Pow(total, 9.0)*float64(maxIterations) - 950.0*math.Pow(total, 99.0)),
					0xff})
			}
		}
	}
}

func mirrorColumnWorker(img *image.RGBA, columnJobs <-chan int, lastColoredRowIndex int, imagPointsLen int, mirrorTop bool) {
	//For every uncolored row we color every pixel
	//the same color as the pixel mirrored in the real axis.
	//(The loop has been switched around for cache awareness.)
	//The +1 in the start value of j comes from how we break the row
	//coloring after having computed the first row after
	//crossing the real axis during the iterations.
	var loops int
	for i := range columnJobs {
		//This loops mirrors a column of pixels across the imaginary axis.
		loops = 0
		for j := lastColoredRowIndex + 1; j < imagPointsLen; j++ {
			img.Set(i, j, img.At(i, lastColoredRowIndex-2-loops))

			loops++
		}

		//If we are mirroring from the top we flipped the imaginary axis
		//earlier and must now flip back.
		if mirrorTop {
			var red uint32 //These are local to each goroutine
			var green uint32
			var blue uint32
			var alpha uint32
			var bottomColorBuffer color.RGBA
			var topColorBuffer color.RGBA
			for j := 0; j < int(imagPointsLen/2); j++ {
				red, green, blue, alpha = img.At(i, j).RGBA()
				bottomColorBuffer = color.RGBA{uint8(red), uint8(green), uint8(blue), uint8(alpha)}
				red, green, blue, alpha = img.At(i, imagPointsLen-j-1).RGBA()
				topColorBuffer = color.RGBA{uint8(red), uint8(green), uint8(blue), uint8(alpha)}
				img.Set(i, j, topColorBuffer)
				img.Set(i, imagPointsLen-j-1, bottomColorBuffer)
			}
		}
	}
}

func displayProgress(jobs chan int, jobSize int, head string) {
	for len(jobs) > 0 {
		fmt.Printf(head+"%.0f%%\r", 100*(1-float64(len(jobs))/float64(jobSize)))
		time.Sleep(100 * time.Millisecond)
	}
	fmt.Println(head + "100%") //Well, we did finish. It's not my fault I slept too long and missed the party.
}

//IteratePoint iterates the complex function z_(n+1) = z_n^2 + c
//starting with z_0 = 0 and c = cReal + cImag*i. If the absolute value
//of z exceeds 6, or the number of iterations exceeds maxIterations
//it stops iterating. If the number of iterations == maxIterations it
//returns 0.0, otherwise maxIterations-iterations-4*(abs(z))^(-.4) / float64(maxIterations).
//This is always a number between 0 and 1.
func IteratePoint(cReal float64, cImag float64, maxIterations int) float64 {

	cImagSqr := cImag * cImag
	magSqr := cReal*cReal + cImagSqr

	//Determine if the complex number is in the main cardioid or period two bulb,
	//if so we can instantly return 0.
	if math.Pow(cReal+1, 2)+cImagSqr <= 0.0625 || magSqr*(8.0*magSqr-3.0) <= 0.09375-cReal {
		return 0.0
	}

	//Initialize variables
	var zReal float64
	var zImag float64
	var zRealSqr float64
	var zImagSqr float64
	var iterations int //Initialization sets variables to 0.

	//Iterates the mandelbrot function.
	//This loop has only three multiplications, which is the minimum.
	for iterations < maxIterations && zRealSqr+zImagSqr <= 36 {
		zImag = zReal * zImag
		zImag = zImag + zImag
		zImag = zImag + cImag
		zReal = zRealSqr - zImagSqr + cReal
		zRealSqr = zReal * zReal
		zImagSqr = zImag * zImag
		iterations++
	}

	if iterations == maxIterations {
		return 0.0
	}

	return (float64(maxIterations-iterations) - 4.0*math.Pow(math.Sqrt(zRealSqr+zImagSqr), -0.4)) / float64(maxIterations)
}
