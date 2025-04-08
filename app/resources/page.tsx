import { ResourceAllocation } from "@/components/dashboard/resource-allocation"
import { MetricCard } from "@/components/dashboard/metric-card"
import { Calendar, Clock, DollarSign, Users } from "lucide-react"

export default function ResourcesPage() {
  return (
    <div className="flex flex-col">
      <div className="border-b">
        <div className="flex h-16 items-center px-4">
          <h1 className="text-lg font-semibold">Resource Allocation Planning</h1>
        </div>
      </div>
      <div className="flex-1 space-y-4 p-4 md:p-8 pt-6">
        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
          <MetricCard
            title="Estimated Savings"
            value="$245,000"
            description="Projected annual savings"
            icon={<DollarSign className="h-4 w-4" />}
            trend={{ value: 15, isPositive: true }}
          />
          <MetricCard
            title="Staff Required"
            value="12"
            description="Additional staff needed"
            icon={<Users className="h-4 w-4" />}
            trend={{ value: 3, isPositive: true }}
          />
          <MetricCard
            title="Follow-ups Needed"
            value="65"
            description="Weekly follow-ups"
            icon={<Calendar className="h-4 w-4" />}
            trend={{ value: 8, isPositive: true }}
          />
          <MetricCard
            title="Response Time"
            value="24 hrs"
            description="Target response time"
            icon={<Clock className="h-4 w-4" />}
            trend={{ value: 10, isPositive: false }}
          />
        </div>
        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
          <ResourceAllocation />
        </div>
      </div>
    </div>
  )
}
