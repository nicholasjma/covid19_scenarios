// Get the modal
if (window.self !== window.top) {
    var modal = window.parent.document.getElementById("myModal");
    var scrollContainer = window.parent.document.getElementsByClassName("img-scroll-container")[0];
} else {
    var modal = document.getElementById("myModal");
    var scrollContainer = document.getElementsByClassName("img-scroll-container")[0];
};

var img_code = ""

// Get the image and insert it inside the modal - use its "alt" text as a caption
if (window.self !== window.top) {
  var modalImg = window.parent.document.getElementById("img01");
  var captionText = window.parent.document.getElementById("caption");
} else {
  var modalImg = document.getElementById("img01");
  var captionText = document.getElementById("caption");
};

function img_click(e) {
  var evt = e ? e : window.event;
  if (evt.stopPropagation) {evt.stopPropagation();}
  else {evt.cancelBubble=true;}
}

function open_image(filename) {
  modal.style.top = parent.window.scrollY
  modal.style.display = "block";
  if (filename.includes("/scenarios") || filename.includes("/all_")) {
    scrollContainer.style["padding-top"] = "50px";
    scrollContainer.style["padding-bottom"] = "100px";
    scrollContainer.style["height"] = "calc(100vh - 150px)";
  } else {
    scrollContainer.style["padding-top"] = "0";
    scrollContainer.style["padding-bottom"] = "0";
    scrollContainer.style["height"] = "100vh";
  }
  modalImg.src = "baseline_hourglass_full_white_48dp.png";
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
if (window.self !== window.top) {
  var span = window.parent.document.getElementsByClassName("modal_close")[0];
  var span2 = window.parent.document.getElementsByClassName("modal_summary")[0];
  var span3 = window.parent.document.getElementsByClassName("modal_forecast")[0];
} else {
  var span = document.getElementsByClassName("modal_close")[0];
  var span2 = document.getElementsByClassName("modal_summary")[0];
  var span3 = document.getElementsByClassName("modal_forecast")[0];
};

// When the user clicks on <span> (x), close the modal
span.onclick = function() { 
  modal.style.display = "none";
}
span2.onclick = switch_summary;
span3.onclick = switch_forecast;
modalImg.onclick = img_click;

modal.onclick = function() {
  modal.style.display = "none";
}


if (window.self !== window.top) {
  window.parent.document.onkeydown = function(evt) {
      evt = evt || window.event;
      if (evt.keyCode == 27) {
          modal.style.display = "none";
      }
  };
} else {
  document.onkeydown = function(evt) {
      evt = evt || window.event;
      if (evt.keyCode == 27) {
          modal.style.display = "none";
      }
  };
}
