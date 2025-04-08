import { type NextRequest, NextResponse } from "next/server"
import { exec } from "child_process"
import { promisify } from "util"
import path from "path"
import fs from "fs"

const execAsync = promisify(exec)

export async function POST(request: NextRequest) {
  try {
    const data = await request.json()

    // Validate input data
    if (!data || !Array.isArray(data.patients)) {
      return NextResponse.json({ error: "Invalid input: Expected an array of patients" }, { status: 400 })
    }

    // Write patient data to a temporary file
    const tempFilePath = path.join(process.cwd(), "temp_patient_data.json")
    fs.writeFileSync(tempFilePath, JSON.stringify(data.patients))

    // Execute the prediction script
    const scriptPath = path.join(process.cwd(), "scripts", "predict.py")
    const { stdout, stderr } = await execAsync(`python ${scriptPath} ${tempFilePath}`)

    if (stderr) {
      console.error("Prediction error:", stderr)
      return NextResponse.json({ error: "Error during prediction" }, { status: 500 })
    }

    // Parse the prediction results
    const predictions = JSON.parse(stdout)

    // Clean up the temporary file
    fs.unlinkSync(tempFilePath)

    return NextResponse.json({ predictions })
  } catch (error) {
    console.error("API error:", error)
    return NextResponse.json({ error: "Internal server error" }, { status: 500 })
  }
}
