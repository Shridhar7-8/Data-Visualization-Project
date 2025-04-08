"use client"

import { Chart, ChartContainer, ChartTooltip, ChartTooltipContent } from "@/components/ui/chart"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Area, AreaChart, ResponsiveContainer, XAxis, YAxis } from "recharts"

const data = [
  { month: "Jan", readmissions: 65, predicted: 62 },
  { month: "Feb", readmissions: 59, predicted: 60 },
  { month: "Mar", readmissions: 80, predicted: 78 },
  { month: "Apr", readmissions: 81, predicted: 80 },
  { month: "May", readmissions: 56, predicted: 58 },
  { month: "Jun", readmissions: 55, predicted: 56 },
  { month: "Jul", readmissions: 40, predicted: 42 },
  { month: "Aug", readmissions: 50, predicted: 52 },
  { month: "Sep", readmissions: 70, predicted: 68 },
  { month: "Oct", readmissions: 65, predicted: 66 },
  { month: "Nov", readmissions: 60, predicted: 61 },
  { month: "Dec", readmissions: 70, predicted: 69 },
]

export function ReadmissionTrend({ className }: { className?: string }) {
  return (
    <Card className={className || "col-span-4"}>
      <CardHeader>
        <CardTitle>Readmission Trends</CardTitle>
        <CardDescription>Actual vs. predicted readmissions over the past 12 months</CardDescription>
      </CardHeader>
      <CardContent className="pl-2">
        <Chart>
          <ChartContainer className="h-[300px]">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart
                data={data}
                margin={{
                  top: 5,
                  right: 30,
                  left: 20,
                  bottom: 5,
                }}
              >
                <defs>
                  <linearGradient id="colorReadmissions" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#10b981" stopOpacity={0.8} />
                    <stop offset="95%" stopColor="#10b981" stopOpacity={0} />
                  </linearGradient>
                  <linearGradient id="colorPredicted" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#6366f1" stopOpacity={0.8} />
                    <stop offset="95%" stopColor="#6366f1" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <XAxis dataKey="month" tick={{ fontSize: 12 }} tickLine={false} />
                <YAxis tick={{ fontSize: 12 }} tickLine={false} axisLine={false} />
                <ChartTooltip content={<ChartTooltipContent />} />
                <Area
                  type="monotone"
                  dataKey="readmissions"
                  stroke="#10b981"
                  fillOpacity={1}
                  fill="url(#colorReadmissions)"
                  name="Actual"
                />
                <Area
                  type="monotone"
                  dataKey="predicted"
                  stroke="#6366f1"
                  fillOpacity={1}
                  fill="url(#colorPredicted)"
                  name="Predicted"
                />
              </AreaChart>
            </ResponsiveContainer>
          </ChartContainer>
        </Chart>
      </CardContent>
    </Card>
  )
}
