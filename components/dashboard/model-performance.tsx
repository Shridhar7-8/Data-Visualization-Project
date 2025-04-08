"use client"

import { Chart, ChartContainer, ChartTooltip, ChartTooltipContent } from "@/components/ui/chart"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Line, LineChart, ResponsiveContainer, XAxis, YAxis } from "recharts"

const data = [
  { threshold: 0.1, precision: 0.35, recall: 0.95, f1: 0.51 },
  { threshold: 0.2, precision: 0.42, recall: 0.9, f1: 0.57 },
  { threshold: 0.3, precision: 0.51, recall: 0.85, f1: 0.64 },
  { threshold: 0.4, precision: 0.6, recall: 0.78, f1: 0.68 },
  { threshold: 0.5, precision: 0.68, recall: 0.7, f1: 0.69 },
  { threshold: 0.6, precision: 0.75, recall: 0.62, f1: 0.68 },
  { threshold: 0.7, precision: 0.82, recall: 0.52, f1: 0.64 },
  { threshold: 0.8, precision: 0.88, recall: 0.38, f1: 0.53 },
  { threshold: 0.9, precision: 0.95, recall: 0.2, f1: 0.33 },
]

export function ModelPerformance() {
  return (
    <Card className="col-span-4">
      <CardHeader>
        <CardTitle>Model Performance Metrics</CardTitle>
        <CardDescription>Precision, recall, and F1 score at different prediction thresholds</CardDescription>
      </CardHeader>
      <CardContent className="pl-2">
        <Chart>
          <ChartContainer className="h-[300px]">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart
                data={data}
                margin={{
                  top: 5,
                  right: 30,
                  left: 20,
                  bottom: 5,
                }}
              >
                <XAxis
                  dataKey="threshold"
                  tick={{ fontSize: 12 }}
                  tickLine={false}
                  label={{ value: "Threshold", position: "insideBottom", offset: -5 }}
                />
                <YAxis
                  tick={{ fontSize: 12 }}
                  tickLine={false}
                  axisLine={false}
                  domain={[0, 1]}
                  tickFormatter={(value) => `${(value * 100).toFixed(0)}%`}
                />
                <ChartTooltip
                  content={<ChartTooltipContent formatter={(value: number) => [`${(value * 100).toFixed(0)}%`, ""]} />}
                />
                <Line
                  type="monotone"
                  dataKey="precision"
                  stroke="#10b981"
                  strokeWidth={2}
                  dot={{ r: 4 }}
                  name="Precision"
                />
                <Line type="monotone" dataKey="recall" stroke="#6366f1" strokeWidth={2} dot={{ r: 4 }} name="Recall" />
                <Line type="monotone" dataKey="f1" stroke="#f59e0b" strokeWidth={2} dot={{ r: 4 }} name="F1 Score" />
              </LineChart>
            </ResponsiveContainer>
          </ChartContainer>
        </Chart>
      </CardContent>
    </Card>
  )
}
