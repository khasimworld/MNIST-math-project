var model;

async function loadModel() {
    model = await tf.loadGraphModel('TFJS/model.json');
}

//Pre processing step 1-12.
function predictImage() {
    // console.log('processing')

    //step1: Load image
    let image = cv.imread(canvas);//to read image
    
    //Step 2: Covert img to black and white.
    cv.cvtColor(image, image, cv.COLOR_RGBA2GRAY, 0); //source and destination both are same img.
    cv.threshold(image,image,175,255,cv.THRESH_BINARY);//anything above 175 to white255.
    
    //step 3:find the contours(outline) to claculate bounding rectangle.
    let contours = new cv.MatVector();
    let hierarchy = new cv.Mat();
    cv.findContours(image, contours, hierarchy, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE);
    
    //step 4: Calculating the bounding rectangle.
    let cnt = contours.get(0);
    let rect = cv.boundingRect(cnt);
    
    //step 5: Crop the image. Region of Interest.
    image = image.roi(rect);

    // step 6: Calculate new size
    var height = image.rows;
    var width = image.cols;
    if (height>width) {
        height = 20;
        const scaleFactor = image.rows/height;
        width = Math.round(image.cols/scaleFactor);

    } else {
        width = 20;
        const scaleFactor = image.cols/width;
        height = Math.round(image.rows/scaleFactor);
        
    }
    
    let newSize = new cv.Size(width,height);//dsize
        
    // step 7: Resize image
    cv.resize(image,image,newSize,0,0,cv.INTER_AREA);

    // Step 8: Add Padding

    const LEFT = Math.ceil(4 + (20-width)/2);
    const RIGHT = Math.floor(4 + (20-width)/2);
    const TOP =Math.ceil(4 + (20-height)/2);
    const BOTTOM =Math.floor(4 + (20-height)/2);
    // console.log(`top:${TOP}, bottom: ${BOTTOM}, left: ${LEFT}, right:${RIGHT}`);
    
    const BLACK =  new cv.Scalar(0,0,0,0);//color rgb trasprncy.

    cv.copyMakeBorder(image, image, TOP, BOTTOM, LEFT, RIGHT,cv.BORDER_CONSTANT, BLACK); //adding padding

    //step 9: Find the centre of Mass
    cv.findContours(image, contours, hierarchy, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE);
    cnt = contours.get(0);
    const Moments = cv.moments(cnt,false);//it'snot binary image so false.
    
    //weight moments.moo is total mass = area of drawn img. max =400=20*20;
    const cx = Moments.m10/Moments.m00;//centre 14px,14 because total 28px including padding.
    const cy = Moments.m01/Moments.m00;

    // console.log(`M00: ${Moments.m00}, cx: ${cx}, cy: ${cy}`);

    //step 10: Shift the image to the cenre of mass
    const X_SHIFT =Math.round(image.cols/2.0 - cx); //14 is the center
    const Y_SHIFT  = Math.round(image.rows/2.0 - cy);

    newSize = new cv.Size(image.cols,image.rows); //dsize
    const M = cv.matFromArray(2, 3, cv.CV_64FC1, [1, 0, X_SHIFT, 0, 1, Y_SHIFT]);
    cv.warpAffine(image, image, M, newSize, cv.INTER_LINEAR, cv.BORDER_CONSTANT, BLACK);

    //step 11: Normalize the Pizel value
                            
    let pixelValues = image.data; // the values are b/w 0-255, we will change to 0-1
    // console.log(`pixelValues: ${pixelValues}`);

    pixelValues = Float32Array.from(pixelValues);//converting to float
    
                            //for dividing all elements in array
    pixelValues = pixelValues.map(function(item) {
        return item/255.0;  
    });

    // console.log(`scaled array: ${pixelValues}`);
    
    //Step 12: Create a Tensor
    const X = tf.tensor([pixelValues]);//bracket for 2dimentions.1 pair aleready in the pixels.
    // console.log(`shape of tensor ${X.shape}`);
    // console.log(`dtype of tensor ${X.dtype}`);

    // Make prediction
    const result = model.predict(X);
    console.log(`The written values is: ${result}`);
    // console.log(tf.memory());
    
    // saving output predicted value from the Tensor
    const output = result.dataSync()[0];
    


    // //testing only
    // const outputCanvas = document.createElement('CANVAS');
    // cv.imshow(outputCanvas, image);//todisplay image
    // document.body.appendChild(outputCanvas);// adding canvas to body

    //Cleanup
    image.delete();
    contours.delete();
    cnt.delete();
    hierarchy.delete();
    M.delete();
    X.dispose();
    result.dispose();
    
    return output;
}