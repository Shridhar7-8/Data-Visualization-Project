"use client"

import { Chart, ChartContainer, ChartTooltip, ChartTooltipContent } from "@/components/ui/chart"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Bar, BarChart, ResponsiveContainer, XAxis, YAxis } from "recharts"

const data = [
  { name: "Nurse Follow-ups", allocated: 45, recommended: 65 },
  { name: "Telehealth", allocated: 30, recommended: 40 },
  { name: "Home Visits", allocated: 15, recommended: 25 },
  { name: "Medication Review", allocated: 25, recommended: 35 },
  { name: "Care Coordination", allocated: 20, recommended: 30 },
]

export function ResourceAllocation() {
  return (
    <Card className="col-span-4">
      <CardHeader>
        <CardTitle>Resource Allocation Recommendations</CardTitle>
        <CardDescription>Current vs. recommended resource allocation based on risk predictions</CardDescription>
      </CardHeader>
      <CardContent className="pl-2">
        <Chart>
          <ChartContainer className="h-[300px]">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart
                data={data}
                margin={{
                  top: 5,
                  right: 30,
                  left: 20,
                  bottom: 5,
                }}
              >
                <XAxis dataKey="name" tick={{ fontSize: 12 }} tickLine={false} />
                <YAxis tick={{ fontSize: 12 }} tickLine={false} axisLine={false} />
                <ChartTooltip content={<ChartTooltipContent />} />
                <Bar dataKey="allocated" fill="#94a3b8" name="Currently Allocated" radius={[4, 4, 0, 0]} />
                <Bar dataKey="recommended" fill="#10b981" name="Recommended" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </ChartContainer>
        </Chart>
      </CardContent>
    </Card>
  )
}
