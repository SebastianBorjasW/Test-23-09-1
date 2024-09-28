import React, { useState } from "react";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";

interface ClassProbability {
  class: string;
  probability: number;
}
interface ClassificationResult {
  predicted_class: string;
  max_probability: number;
  inference_time: number;
  class_probability: ClassProbability[];
  warning?: string;
}

const ImageClassifier = () => {
  const [image, setImage] = useState<File | null>(null);
  const [result, setResult] = useState<ClassificationResult | null>(null);
  const [isLoading, setIsLoading] = useState(false);

  const handleImageUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    if (event.target.files && event.target.files[0]) {
      setImage(event.target.files[0]);
      setResult(null);
    }
  };

  const classifyImage = async () => {
    if (!image) {
      alert("Carge una imagen");
      return;
    }

    setIsLoading(true);
    try {
      const formData = new FormData();
      formData.append("file", image);

      const response = await fetch("http://localhost:5000/classify", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error("Error en la clasificacion");
      }

      const result = await response.json();
      setResult(result);
    } catch (error) {
      alert("Error al clasificar");
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="flex flex-row gap-4">
      <div
        className="max-w-md mx-auto mt-10 p-6 bg-white rounded-lg shadow-xl"
        style={{ backgroundColor: "#f0f0f0" }}
      >
        <h2 className="text-2xl font-bold mb-4">Clasificador de imagenes</h2>
        <div className="grid w-full max-w-sm items-center gap-1.5">
          <Label htmlFor="picture">Picture</Label>
          <Input
            id="picture"
            type="file"
            accept="image/*"
            onChange={handleImageUpload}
            className="mb-2"
          />
        </div>
        {image && (
          <img
            src={URL.createObjectURL(image)}
            alt="Preview"
            className="w-full max-w-xs h-auto object-contain mx-auto"
          />
        )}
        <br></br>
        <button
          onClick={classifyImage}
          disabled={!image || isLoading}
          className="bg-blue-950 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded disabled:opacity-50"
        >
          {isLoading ? "Clasificando..." : "Clasificar imagen"}
        </button>
        {result && (
          <div className="mt-4">
            <p>
              <strong>Prediccion:</strong> {result.predicted_class}
            </p>
            <p>
              <strong>Probabilidad:</strong>{" "}
              {(result.max_probability * 100).toFixed(2)}%
            </p>
            <p>
              <strong>Tiempo:</strong>
              {result.inference_time.toFixed(4)}segundos
            </p>
            {result.warning && (
              <p className="text-yellow-200">
                <strong>mensaje</strong>
                {result.warning}
              </p>
            )}
          </div>
        )}
      </div>
      <div
        className="max-w-md mx-auto mt-10 p-6 bg-white rounded-lg shadow-xl"
        style={{ backgroundColor: "#f0f0f0" }}
      > 
        <h3 className="text-2xl font-bold mb-4">Probabilidad por clase</h3>
        {result && (
          <div className="mt-4">
            <ul>
              {result.class_probability.map((item, index) => (
                <li key={index}>
                  {item.class}: {item.probability * 100}
                </li>
              ))}
            </ul>
          </div>
        )}
      </div>
    </div>
  );
};

export default ImageClassifier;
