inputImage = document.getElementById("inputImage");
colorInput = document.getElementById("colorInput");
outputImage = document.getElementById("outputImage");

colorInput.addEventListener("focusout", () => {});

function processingEffect(isProcessing) {
  imageInputElement = outputImage;

  isProcessing
    ? (imageInputElement.style.filter = "blur(5px)")
    : (imageInputElement.style.filter = "none");
}

// on input-image click
inputImage.addEventListener("click", () => {
  let fileSelector = document.createElement("input");
  fileSelector.type = "file";
  fileSelector.accept = "image/*";
  fileSelector.style.display = "none";
  document.body.appendChild(fileSelector);

  fileSelector.addEventListener("change", function () {
    const file = fileSelector.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = function (e) {
        const img = inputImage;
        img.src = e.target.result;
        img.style.display = "block";
      };
      reader.readAsDataURL(file);
    }
    document.body.removeChild(fileSelector);
  });

  fileSelector.click();
});

// on convert btn click
document.getElementById("convertBtn").addEventListener("click", async () => {
  processingEffect(true);
  try {
    let response = await fetch(inputImage.src);
    let blob = await response.blob();

    let file = new File([blob], "image.jpg", { type: blob.type });

    let colorCode = colorInput.value;

    let formData = new FormData();

    formData.append("image", file);
    formData.append("color", colorCode);

    let uploadResponse = await fetch("/api/upload", {
      method: "POST",
      body: formData,
    });

    let responseBlob = await uploadResponse.blob();
    let responseSrc = URL.createObjectURL(responseBlob);
    outputImage.src = responseSrc;

    processingEffect(false);
  } catch (error) {
    console.error(error);
  }
});

// on download btn click
document.getElementById("downloadBtn").addEventListener("click", () => {
  const imgElement = outputImage;
  const imageURL = imgElement.src;

  const anchor = document.createElement("a"); // Create a temporary <a> element
  anchor.href = imageURL;
  anchor.download = "output_image.jpg"; // Set the file name
  document.body.appendChild(anchor);
  anchor.click(); // Trigger the download
  document.body.removeChild(anchor);
});
