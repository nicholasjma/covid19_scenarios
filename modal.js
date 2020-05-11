// Get the modal
var modal = parent.document.getElementById("myModal");
var img_code = ""

// Get the image and insert it inside the modal - use its "alt" text as a caption
var modalImg = parent.document.getElementById("img01");
var captionText = parent.document.getElementById("caption");

function open_image(filename) {
  modal.style.display = "block";
  modalImg.src = filename;
}

function switch_forecast(e) {
  modal.style.src.display = "block";
  if (modalImg.src.includes("/summary")) {
    open_image(modalImg.src.replace("/summary", "/scenarios"));
  }
  var evt = e ? e : window.event;
  if (evt.stopPropagation) {evt.stopPropagation();}
  else {evt.cancelBubble=true;}
}

function switch_summary(e) {
  modal.style.display = "block";
  if (modalImg.src.includes("/scenarios")) {
    open_image(modalImg.src.replace("/scenarios", "/summary"));
  }
  var evt = e ? e : window.event;
  if (evt.stopPropagation) {evt.stopPropagation();}
  else {evt.cancelBubble=true;}
}
// Get the <span> element that closes the modal
var span = parent.document.getElementsByClassName("modal_close")[0];
var span2 = parent.document.getElementsByClassName("modal_summary")[0];
var span3 = parent.document.getElementsByClassName("modal_forecast")[0];

// When the user clicks on <span> (x), close the modal
span.onclick = function() { 
  modal.style.display = "none";
}
span2.onclick = switch_summary;
span3.onclick = switch_forecast;

modal.onclick = function() {
  modal.style.display = "none";
}

modalImg.onclick = function() {
}

parent.document.onkeydown = function(evt) {
    evt = evt || window.event;
    if (evt.keyCode == 27) {
        modal.style.display = "none";
    }
};
