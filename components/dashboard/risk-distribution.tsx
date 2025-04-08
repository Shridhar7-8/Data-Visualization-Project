"use client"

import { Chart, ChartContainer, ChartTooltip, ChartTooltipContent } from "@/components/ui/chart"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Cell, Pie, PieChart, ResponsiveContainer } from "recharts"

const data = [
  { name: "Low Risk", value: 540, color: "#10b981" },
  { name: "Medium Risk", value: 320, color: "#f59e0b" },
  { name: "High Risk", value: 140, color: "#ef4444" },
]

export function RiskDistribution({ className }: { className?: string }) {
  return (
    <Card className={className || "col-span-2"}>
      <CardHeader>
        <CardTitle>Patient Risk Distribution</CardTitle>
        <CardDescription>Current patient population by readmission risk level</CardDescription>
      </CardHeader>
      <CardContent>
        <Chart>
          <ChartContainer className="h-[300px]">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={data}
                  cx="50%"
                  cy="50%"
                  innerRadius={60}
                  outerRadius={80}
                  paddingAngle={5}
                  dataKey="value"
                  nameKey="name"
                >
                  {data.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Pie>
                <ChartTooltip content={<ChartTooltipContent />} />
              </PieChart>
            </ResponsiveContainer>
          </ChartContainer>
        </Chart>
        <div className="mt-4 grid grid-cols-3 gap-4 text-center">
          {data.map((item) => (
            <div key={item.name} className="flex flex-col items-center">
              <div className="flex items-center">
                <div className="mr-1 h-3 w-3 rounded-full" style={{ backgroundColor: item.color }} />
                <span className="text-sm font-medium">{item.name}</span>
              </div>
              <span className="text-sm text-muted-foreground">{item.value} patients</span>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  )
}
