"use client"

import { Chart, ChartContainer, ChartTooltip, ChartTooltipContent } from "@/components/ui/chart"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { BarChart, ResponsiveContainer, XAxis, YAxis } from "recharts"

const data = [
  { name: "Medication Adherence", value: 0.85 },
  { name: "Heart Failure", value: 0.78 },
  { name: "COPD", value: 0.72 },
  { name: "Age > 75", value: 0.68 },
  { name: "Previous Admissions", value: 0.65 },
  { name: "Length of Stay", value: 0.62 },
  { name: "Diabetes", value: 0.58 },
  { name: "Discharge Planning", value: 0.52 },
]

export function FactorImportance({ className }: { className?: string }) {
  return (
    <Card className={className || "col-span-2"}>
      <CardHeader>
        <CardTitle>Key Readmission Factors</CardTitle>
        <CardDescription>Feature importance in the prediction model</CardDescription>
      </CardHeader>
      <CardContent className="pl-2">
        <Chart>
          <ChartContainer className="h-[300px]">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart
                data={data}
                layout="vertical"
                margin={{
                  top: 5,
                  right: 30,
                  left: 90,
                  bottom: 5,
                }}
              >
                <XAxis
                  type="number"
                  domain={[0, 1]}
                  tickFormatter={(value) => `${(value * 100).toFixed(0)}%`}
                  tick={{ fontSize: 12 }}
                />
                <YAxis dataKey="name" type="category" tick={{ fontSize: 12 }} width={80} />
                <ChartTooltip
                  content={
                    <ChartTooltipContent
                      formatter={(value: number) => [`${(value * 100).toFixed(0)}%`, "Importance"]}
                    />
                  }
                />
              </BarChart>
            </ResponsiveContainer>
          </ChartContainer>
        </Chart>
      </CardContent>
    </Card>
  )
}
