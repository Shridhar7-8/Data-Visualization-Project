import { BarChart3, Calendar, Clock, Users } from "lucide-react"

import { MetricCard } from "@/components/dashboard/metric-card"
import { ReadmissionTrend } from "@/components/dashboard/readmission-trend"
import { RiskDistribution } from "@/components/dashboard/risk-distribution"
import { FactorImportance } from "@/components/dashboard/factor-importance"
import { PatientTable } from "@/components/dashboard/patient-table"
import { ModelPerformance } from "@/components/dashboard/model-performance"

export default function Dashboard() {
  return (
    <div className="flex flex-col">
      <div className="border-b">
        <div className="flex h-16 items-center px-4">
          <h1 className="text-lg font-semibold">Hospital Readmission Dashboard</h1>
        </div>
      </div>
      <div className="flex-1 space-y-4 p-4 md:p-8 pt-6">
        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
          <MetricCard
            title="Total Patients"
            value="1,000"
            description="Currently monitored patients"
            icon={<Users className="h-4 w-4" />}
            trend={{ value: 12, isPositive: true }}
          />
          <MetricCard
            title="Readmission Rate"
            value="14.2%"
            description="Last 30 days"
            icon={<Calendar className="h-4 w-4" />}
            trend={{ value: 2.5, isPositive: false }}
          />
          <MetricCard
            title="High Risk Patients"
            value="140"
            description="Requiring intervention"
            icon={<BarChart3 className="h-4 w-4" />}
            trend={{ value: 8, isPositive: false }}
          />
          <MetricCard
            title="Avg. Time to Readmission"
            value="18 days"
            description="For readmitted patients"
            icon={<Clock className="h-4 w-4" />}
            trend={{ value: 1, isPositive: true }}
          />
        </div>
        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
          <ReadmissionTrend />
        </div>
        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
          <RiskDistribution />
          <FactorImportance />
        </div>
        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
          <ModelPerformance />
        </div>
        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
          <PatientTable />
        </div>
      </div>
    </div>
  )
}
