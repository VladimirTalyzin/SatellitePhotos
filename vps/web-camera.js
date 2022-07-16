class WebCamera
{
	constructor(webCameraElement, facingMode = "user", width, height)
	{
		this._webCameraElement = webCameraElement

		this._webCameraElement.width = (typeof width == "number") && width > 0 ? width : 640
		this._webCameraElement.height = (typeof height == "number") && height > 0 ? height : this._webCameraElement.offsetWidth * (3 / 4)

		this._facingMode = facingMode
		this._webCamerasList = []
		this._streamsList = []
		this._selectedDeviceID = ""
		this._canvasElement = null
	}

	async start(startStream = true)
	{
		return new Promise((resolve, reject) =>
		{
			this.stop()
			navigator.mediaDevices.getUserMedia(this.__getMediaConstraints()).then(stream =>
			{
				this._streamsList.push(stream)
				this.__info().then(() =>
				{
					this.__selectFace(this._facingMode)
					if (startStream)
					{
						this.__stream()
							.then(() =>
							{
								resolve(this._facingMode)
							})
							.catch(error =>
							{
								reject(error)
							})
					}
					else
					{
						resolve(this._selectedDeviceID)
					}
				})
				.catch(error =>
				{
					reject(error)
				})
			})
			.catch(error =>
			{
				reject(error)
			})
		})
	}

	flip()
	{
		this._facingMode = (this._facingMode == "user") ? "environment" : "user"
		this._webCameraElement.style.transform = ""
		this.__selectFace(this._facingMode)
		this.start().then()
	}

	getPhoto()
	{
		if (this._canvasElement == null)
		{
			this._canvasElement = document.createElement("canvas")
			this._canvasElement.style.display = "none"
			document.body.appendChild(this._canvasElement)
		}

		this._canvasElement.height = this._webCameraElement.scrollHeight
		this._canvasElement.width = this._webCameraElement.scrollWidth
		let context = this._canvasElement.getContext("2d")

		if (this._facingMode == "user")
		{
			context.translate(this._canvasElement.width, 0)
			context.scale(-1, 1)
		}

		context.clearRect(0, 0, this._canvasElement.width, this._canvasElement.height)
		context.drawImage(this._webCameraElement, 0, 0, this._canvasElement.width, this._canvasElement.height)
		return this._canvasElement.toDataURL("image/png")
	}

	stop()
	{
		this._streamsList.forEach(stream =>
		{
			stream.getTracks().forEach(track =>
			{
				track.stop()
			})
		})
	}


	__getVideoInputs(mediaDevices)
	{
		this._webCamerasList = []
		mediaDevices.forEach(mediaDevice =>
		{
			// noinspection SpellCheckingInspection
			if (mediaDevice.kind === "videoinput")
			{
				this._webCamerasList.push(mediaDevice)
			}
		})
		if (this._webCamerasList.length == 1)
		{
			this._facingMode = "user"
		}
		return this._webCamerasList
	}
	
	__getMediaConstraints()
	{
		const videoConstraints = {}
		if (this._selectedDeviceID == "")
		{
			videoConstraints.facingMode = this._facingMode
		}
		else
		{
			videoConstraints.deviceId = {exact: this._selectedDeviceID}
		}
		
		return {
			video: videoConstraints,
			audio: false,
		}
	}

    __selectFace(face)
    {
        for (let webcam of this._webCamerasList)
        {
            if ((face == "user" && webcam.label.toLowerCase().includes("front")) ||
                (face == "environment" && webcam.label.toLowerCase().includes("back")))
            {
                this._selectedDeviceID = webcam.deviceId
                break
            }
        }
    }

	async __info()
	{
		return new Promise((resolve, reject) =>
		{
			navigator.mediaDevices.enumerateDevices().then(devices =>
			{
				this.__getVideoInputs(devices)
				resolve(this._webCamerasList)
			})
			.catch(error =>
			{
				reject(error)
			})
		})
	}

	async __stream()
	{
		return new Promise((resolve, reject) =>
		{
			navigator.mediaDevices.getUserMedia(this.__getMediaConstraints()).then(stream =>
			{
				this._streamsList.push(stream)
				this._webCameraElement.srcObject = stream
				if (this._facingMode == "user")
				{
					this._webCameraElement.style.transform = "scale(-1, 1)"
				}
				this._webCameraElement.play()
				resolve(this._facingMode)
			})
			.catch(error =>
			{
				console.log(error)
				reject(error)
			})
		})
	}
}