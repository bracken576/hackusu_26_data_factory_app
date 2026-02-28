# Approval Workflow

New dashboard templates follow this lifecycle:

```
Developer submits → IT review (guardrail check) → Sandbox deploy → Approval → Production
```

- Sandbox apps are flagged with `category: "dev"` in the admin page
- Production apps are promoted by setting `category: "main"`
- The admin dashboard shows pending approvals and deployment status
