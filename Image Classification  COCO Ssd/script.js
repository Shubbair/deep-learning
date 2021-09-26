 const demosSection = document.getElementById('demos');

 var model = undefined;
 
 // Before we can use COCO-SSD class we must wait for it to finish
 // loading. Machine Learning models can be large and take a moment to
 // get everything needed to run.
 cocoSsd.load().then(function (loadedModel) {
   model = loadedModel;
   // Show demo section now model is ready to use.
   demosSection.classList.remove('invisible');
 });
 
 
 /********************************************************************
 // Demo 1: Grab a bunch of images from the page and classify them
 // upon click.
 ********************************************************************/
 
 // In this demo, we have put all our clickable images in divs with the 
 // CSS class 'classifyOnClick'. Lets get all the elements that have
 // this class.
 const imageContainers = document.getElementsByClassName('classifyOnClick');
 
 // Now let's go through all of these and add a click event listener.
 for (let i = 0; i < imageContainers.length; i++) {
   // Add event listener to the child element whichis the img element.
   imageContainers[i].children[0].addEventListener('click', handleClick);
 }
 
 // When an image is clicked, let's classify it and display results!
 function handleClick(event) {
   if (!model) {
     console.log('Wait for model to load before clicking!');
     return;
   }
   // We can call model.classify as many times as we like with
   // different image data each time. This returns a promise
   // which we wait to complete and then call a function to
   // print out the results of the prediction.
   model.detect(event.target).then(function (predictions) {
     // Lets write the predictions to a new paragraph element and
     // add it to the DOM.
     console.log(predictions);
     for (let n = 0; n < predictions.length; n++) {
       // Description text
       const p = document.createElement('p');
       p.innerText = predictions[n].class  + ' - with ' 
           + Math.round(parseFloat(predictions[n].score) * 100) 
           + '% confidence.';
       // Positioned at the top left of the bounding box.
       // Height is whatever the text takes up.
       // Width subtracts text padding in CSS so fits perfectly.
       p.style = 'left: ' + predictions[n].bbox[0] + 'px;' + 
           'top: ' + predictions[n].bbox[1] + 'px; ' + 
           'width: ' + (predictions[n].bbox[2] - 10) + 'px;';
 
       const highlighter = document.createElement('div');
       highlighter.setAttribute('class', 'highlighter');
       highlighter.style = 'left: ' + predictions[n].bbox[0] + 'px;' +
           'top: ' + predictions[n].bbox[1] + 'px;' +
           'width: ' + predictions[n].bbox[2] + 'px;' +
           'height: ' + predictions[n].bbox[3] + 'px;';
 
       event.target.parentNode.appendChild(highlighter);
       event.target.parentNode.appendChild(p);
     }
   });
 }