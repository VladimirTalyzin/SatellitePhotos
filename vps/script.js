document.addEventListener("DOMContentLoaded", () =>
{
	document.getElementById("select-file").addEventListener("change", async (event) =>
	{
		const buffer = await event.target.files[0].arrayBuffer()
		const postData = new FormData()
		postData.append("photo", ";," + btoa(String.fromCharCode(...new Uint8Array(buffer))))

		postImage(postData)
	}, false)

	const webCameraPlace = document.getElementById("web-camera")

	const screenWidth = window.innerWidth
	const screeHeight = window.innerHeight
	let width
	let height

	if (screenWidth > screeHeight)
	{
		width = Math.round(4 / 3 * screeHeight / 2)
		height = Math.round(screeHeight / 2)
	}
	else
	{
		width = screenWidth
		height = Math.round(screenWidth * 3 / 4)
	}

	webCameraPlace.style.width = width + "px"
	webCameraPlace.style.height = height + "px"

	const webCamera = new WebCamera(webCameraPlace, "environment", width, height)

	webCamera.flip()

	webCamera.start().then(() =>
	{
		console.log("Camera started")
	})
	.catch((exception) =>
	{
		console.log(exception)
	})

	document.getElementById("switch-face").addEventListener("click", () =>
	{
		webCamera.flip()
	})

	let mirror = true
	document.getElementById("mirror").addEventListener("click", () =>
	{
		mirror = !mirror
		if (mirror)
		{
			webCameraPlace.style.transform = "scaleX(-1)"
		}
		else
		{
			webCameraPlace.style.transform = null
		}
	})

	const getPhoto = document.getElementById("get-photo")
	getPhoto.addEventListener("click", () =>
	{
		const postData = new FormData()
		postData.append("photo", webCamera.getPhoto())
		const saveText = getPhoto.innerHTML
		getPhoto.innerHTML = "⟳"
		getPhoto.style.pointerEvents = "none"

		postImage(postData, () =>
		{
			getPhoto.style.pointerEvents = null
			getPhoto.innerHTML = saveText
		})
	})
})

function postImage(postData, onAfterUpload)
{
	document.documentElement.style.cursor = "wait"
	fetch("https://0v.ru/satellite/detect.php", {method: "POST", body: postData})
		.then(response => response.blob())
		.then(imageBlob =>
		{
			// Then create a local URL for that image and print it
			const imageObjectURL = URL.createObjectURL(imageBlob)
			const resultImage = document.createElement("img")
			resultImage.style.textAlign = "center"
			resultImage.style.position = "fixed"
			resultImage.style.top = "10pt"
			resultImage.style.maxWidth = "90%"
			resultImage.addEventListener("click", () => resultImage.remove())
			document.body.appendChild(resultImage)

			resultImage.src = imageObjectURL
			console.log(imageObjectURL)
			document.documentElement.style.cursor = null

			if (typeof(onAfterUpload) == "function")
			{
				onAfterUpload()
			}
		})
		.catch()
		{
			if (typeof(onAfterUpload) == "function")
			{
				onAfterUpload()
			}
		}
}

function selectTest(callTestImage)
{
	const postData = new FormData()
	postData.append("photo", getBase64Image(callTestImage))

	postImage(postData)

	return false
}

function getBase64Image(testImage)
{
	const canvas = document.createElement("canvas")
	canvas.width = testImage.width
	canvas.height = testImage.height
	const context = canvas.getContext("2d")
	context.drawImage(testImage, 0, 0)
	return canvas.toDataURL("image/png")
}